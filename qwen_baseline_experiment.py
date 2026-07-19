import modal

# 1. Define the environment (Image)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "pandas",
        "tqdm",
        "torch",
        "transformers",
        "scikit-learn",
        "datasets",
        "matplotlib",
        "nltk",
        "tiktoken",
        "lightning",
        "faiss-gpu",
        "accelerate" 
    )
)

# 2. Reference the persistent volume
vol = modal.Volume.from_name("ai-safety", create_if_missing=True)

# 3. Define the Modal App
app = modal.App("ai-safety-baseline")

# 4. The main execution function
@app.function(
    image=image, 
    volumes={"/mnt/ai-safety": vol}, 
    gpu="A10G", 
    timeout=86400,  
    secrets=[modal.Secret.from_name("huggingface-secret")] 
)
def run_baseline_generation():
    import os
    import sys
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from scipy.stats import norm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lightning import Fabric

    print("Starting baseline environment setup...")

    # CRITICAL: Change directory so the detector's local 'src' modules can be imported
    os.chdir("/mnt/ai-safety/ood-llm-detect/")
    sys.path.append("/mnt/ai-safety/ood-llm-detect/")

    # --- CONFIGURATION ---
    INPUT_RESULTS_CSV = "/mnt/ai-safety/constraint_search_results_full.csv"
    OUTPUT_BASELINE_CSV = "/mnt/ai-safety/humanized_baseline.csv"

    # --- DETECTOR SETUP ---
    distrib_params = {
        'deepfake': {'mu0': 2.8207, 'sigma0': 1.188, 'mu1': 0.2149, 'sigma1': 2.3777},
        'M4': {'mu0': 2.8210, 'sigma0': 1.3977, 'mu1': 0.08976, 'sigma1': 2.79554},
        'raid': {'mu0': 3.3258, 'sigma0': 1.19811, 'mu1': 0.2563, 'sigma1': 2.39623}
    }

    def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
        pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
        pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
        prob = pdf_value1 / (pdf_value0 + pdf_value1)
        return prob

    @torch.no_grad()
    def predict_single(text, model, tokenizer, device="cuda", dataset_name="deepfake"):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        model.eval()
        loss, out, _, _ = model(encoded, 0, 0, torch.tensor([0]).to(device))
        prob = compute_prob_norm(out.cpu().numpy(),
                        distrib_params[dataset_name]['mu0'], distrib_params[dataset_name]['sigma0'],
                        distrib_params[dataset_name]['mu1'], distrib_params[dataset_name]['sigma1'])
        return prob.item()

    def load_dsvdd_model(opt):
        if opt.ood_type == "deepsvdd":
            from src.deep_SVDD import SimCLR_Classifier_SCL
        elif opt.ood_type == "energy":
            from src.energy import SimCLR_Classifier_SCL
        elif opt.ood_type == "hrn":
            from src.hrn import SimCLR_Classifier_SCL
        else:
            raise ValueError("Only support deepsvdd, hrn and energy")

        fabric = Fabric(accelerator="cuda", devices=1)
        fabric.launch()
        if opt.ood_type == "hrn":
            model = SimCLR_Classifier_SCL(opt, opt.num_models, fabric)
        else:
            model = SimCLR_Classifier_SCL(opt, fabric)
        
        state_dict = torch.load(opt.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model = model.cuda()
        tokenizer = model.model.tokenizer
        return model, tokenizer

    def make_opt(device_num=8, temperature=0.07, a=1.0, d=1.0, nu=0.1, objective="one-class", out_dim=128, only_classifier=False, mode="deepfake", ood_type="deepsvdd", model_path="", model_name="princeton-nlp/unsup-simcse-roberta-base"):
        return SimpleNamespace(device_num=device_num, temperature=temperature, a=a, d=d, nu=nu, objective=objective, out_dim=out_dim, only_classifier=only_classifier, mode=mode, ood_type=ood_type, model_path=model_path, model_name=model_name)

    print("Loading Detector Model...")
    opt_deepfake = make_opt(
        model_path="/mnt/ai-safety/ood-llm-detect/dsvdd_deepfake.pth",
        ood_type="deepsvdd",
        mode="deepfake",
        out_dim=768
    )
    dsvdd_deepfake_model, dsvdd_deepfake_tokenizer = load_dsvdd_model(opt_deepfake)

    def detector_adapter(text):
        return predict_single(text, dsvdd_deepfake_model, dsvdd_deepfake_tokenizer, dataset_name=opt_deepfake.mode)

    # --- GENERATOR SETUP ---
    print("Loading Qwen Generator Model...")
    model_name = "Qwen/Qwen3.5-9B"
    qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )

    def qwen_generate(prompt, enable_thinking=False, max_new_tokens=2048):
        messages = [{"role": "user", "content": prompt}]
        text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
        generated_ids = qwen_model.generate(**model_inputs, max_new_tokens=max_new_tokens, pad_token_id=qwen_tokenizer.eos_token_id)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = qwen_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content

    # --- HUMANIZER PROMPT ---
    humanize_prompt_template = (
        "Rewrite the following text to sound more human, natural, and emotionally resonant. "
        "Avoid repetitive patterns, robotic phrasing, or excessive structure. "
        "Maintain the core message and meaning, and keep the length roughly the same (around 350 words).\n\n"
        "Original Text:\n{TEXT}\n\n"
        "Humanized Version:"
    )

    def humanize_text(raw_text):
        filled = humanize_prompt_template.format(TEXT=raw_text)
        humanized = qwen_generate(filled, enable_thinking=False, max_new_tokens=2048)
        return humanized if humanized is not None else raw_text


    # --- PROCESSING AND ORCHESTRATION ---
    if not os.path.exists(INPUT_RESULTS_CSV):
        raise FileNotFoundError(f"Could not find the previous results file at {INPUT_RESULTS_CSV}")

    df = pd.read_csv(INPUT_RESULTS_CSV)
    print(f"Loaded {len(df)} total rows from search results.")

    # Filter down to strictly Depth 0 results (your initial raw generations)
    depth_0_df = df[df['depth'] == 0].copy()
    print(f"Isolated {len(depth_0_df)} raw depth 0 baselines to process.")

    # Checkpointing: Allow resuming if this baseline run stops mid-way
    completed_ids = set()
    if os.path.exists(OUTPUT_BASELINE_CSV):
        existing_df = pd.read_csv(OUTPUT_BASELINE_CSV)
        if 'prompt_id' in existing_df.columns:
            completed_ids = set(existing_df['prompt_id'].unique())
        results = existing_df.to_dict('records')
        print(f"Found existing baseline output file. Resuming. {len(completed_ids)} prompts already humanized.")
    else:
        results = []

    for idx, row in depth_0_df.iterrows():
        prompt_id = row.get('prompt_id')

        # Skip if already humanized in a previous run
        if prompt_id in completed_ids:
            print(f"Skipping prompt_id={prompt_id}, already processed.")
            continue

        print(f"\n=== Humanizing baseline for prompt id={prompt_id} ===")
        original_raw_text = str(row['generated_text'])

        # 1. Humanize using Qwen
        humanized_text = humanize_text(original_raw_text)

        # 2. Get the detector score for the humanized copy
        humanized_score = detector_adapter(humanized_text)

        # 3. Append detailed side-by-side metrics
        results.append({
            "prompt_id": prompt_id,
            "prompt_type": row.get('prompt_type'),
            "base_prompt": row.get('prompt_with_constraints'),
            "original_raw_text": original_raw_text,
            "original_detector_score": row.get('detector_score'),
            "humanized_text": humanized_text,
            "humanized_detector_score": humanized_score
        })

        # Save and commit instantly to keep progress safe on the Volume
        pd.DataFrame(results).to_csv(OUTPUT_BASELINE_CSV, index=False)
        vol.commit()
        print(f"Saved checkpoint to {OUTPUT_BASELINE_CSV} (Original Score: {row.get('detector_score'):.4f} -> Humanized Score: {humanized_score:.4f})")

    print(f"\nAll baseline generations have been humanized and saved to {OUTPUT_BASELINE_CSV}!")


# Entry point to trigger the Modal deployment
@app.local_entrypoint()
def main():
    print("Submitting baseline job to Modal...")
    run_baseline_generation.remote()
    print("Baseline job completed!")
