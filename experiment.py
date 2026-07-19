import modal

# 1. Define the environment (Image)
# We pre-install all the pip packages here.
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
        "accelerate" # Added: required for device_map="auto"
    )
)

# 2. Reference the persistent volume
vol = modal.Volume.from_name("ai-safety", create_if_missing=True)

# 3. Define the Modal App
app = modal.App("ai-safety-experiment")

# 4. The main execution function
# We request an A10G GPU and give it a massive timeout (24 hours) for the beam search.
@app.function(
    image=image, 
    volumes={"/mnt/ai-safety": vol}, 
    gpu="A10G", 
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")] # Remove this line if you didn't set up the HF secret
)
def run_experiment():
    import os
    import sys
    import pandas as pd
    import time
    import re
    import torch
    from types import SimpleNamespace
    from scipy.stats import norm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lightning import Fabric

    print("Starting experiment environment setup...")

    # CRITICAL: Change directory so the detector's local 'src' modules can be imported
    os.chdir("/mnt/ai-safety/ood-llm-detect/")
    sys.path.append("/mnt/ai-safety/ood-llm-detect/")

    # --- CONFIGURATION ---
    BEAM_WIDTH = 3
    MAX_DEPTH = 5
    TARGET_LENGTH = 350
    BASE_PROMPTS_CSV = "/mnt/ai-safety/base_prompts_with_categories.csv"
    OUTPUT_CSV = "/mnt/ai-safety/constraint_search_results.csv"

    # --- DEFINITIONS ---
    CATEGORY_DEFINITIONS = {
        "Structural Constraints": "Structural constraints modify the overall shape, organization, and pacing of the text, including how ideas are ordered, how sentence lengths vary across the text, how sections differ in density or elaboration, and how the rhythm or flow changes from beginning to end, without prescribing specific content or wording.",
        "Lexical Constraints": "Lexical constraints modify the overall choice, distribution, and variation of words used in the text, including preferences for certain types of vocabulary, avoidance of particular lexical patterns, or shifts in word frequency, register, or specificity, without dictating the underlying content or structure.",
        "Stylistic Constraints": "Stylistic constraints shape the general tone, voice, and expressive character of the text, influencing how the writing feels or sounds—such as its formality, emotional coloration, narrative attitude, or rhetorical flavor—without prescribing specific words or altering the factual content.",
        "Mimicry Constraints": "Mimicry constraints direct the text to adopt the characteristic style, voice, or expressive tendencies of a particular author, persona, group, or dialect, guiding the overall stylistic imitation without requiring specific phrases or content from the source being mimicked.",
        "Content-Level Constraints": "Content‑level constraints influence the distribution, emphasis, and balance of information across the text, shaping which aspects are expanded, minimized, prioritized, or reframed, without specifying the exact wording or stylistic execution.",
        "Imperfection Constraints": "Imperfection constraints introduce controlled irregularities that emulate natural human variability—such as slight inconsistencies, mild redundancies, subtle awkwardness, or uneven phrasing—without compromising overall clarity or altering the intended meaning of the text."
    }

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

    # --- PROMPT TEMPLATES ---
    generation_prompt_template = (
        "{PROMPT_WITH_CONSTRAINTS}\n\n"
        "Please produce a coherent text of approximately {TARGET_LENGTH} words. "
        "Natural variation is allowed but keep the overall length comparable to this target."
    )

    constraint_prompt_template = (
        "The following prompt was given to an LLM:\n"
        "{CURRENT_PROMPT}\n\n"
        "The LLM produced the following output:\n"
        "{MODEL_OUTPUT}\n\n"
        "An AI-text detector assigned the following probability that this output is AI-generated:\n"
        "{DETECTOR_SCORE}\n\n"
        "Your task is to propose one new constraint that will be added to the original prompt in order to reduce the AI-generated probability assigned by the detector.\n\n"
        "The constraint must come from the following category: {CONSTRAINT_CATEGORY}.\n"
        "This category is defined as: {CATEGORY_DEFINITION}.\n\n"
        "Please output one single sentence that contains only the invented constraint, with no explanation, no justification, and no additional text."
    )

    def generation_adapter(prompt_with_constraints, target_length):
        filled = generation_prompt_template.format(PROMPT_WITH_CONSTRAINTS=prompt_with_constraints, TARGET_LENGTH=target_length)
        gen = qwen_generate(filled, enable_thinking=False, max_new_tokens=2048)
        return gen if gen is not None else ""

    def constraint_generator_adapter(current_prompt, model_output, detector_score, category, category_definition, target_length):
        filled = constraint_prompt_template.format(
            CURRENT_PROMPT=current_prompt, MODEL_OUTPUT=model_output, DETECTOR_SCORE=f"{detector_score:.4f}",
            CONSTRAINT_CATEGORY=category, CATEGORY_DEFINITION=category_definition, TARGET_LENGTH=target_length
        )
        raw = qwen_generate(filled, enable_thinking=False, max_new_tokens=128)
        if raw is None: return ""
        for line in raw.splitlines():
            s = line.strip()
            if s:
                m = re.split(r'(?<=[\.\!\?])\s+', s)
                if len(m) > 0: return m[0].strip()
                return s
        return raw.strip()

    def append_constraint_to_prompt(current_prompt: str, constraint_sentence: str) -> str:
        if not constraint_sentence: return current_prompt
        return current_prompt.strip() + "\n" + constraint_sentence.strip()

    def lambda_score(node):
        return node["detector_score"]

    # --- BEAM SEARCH LOGIC ---
    def run_constraint_beam_search_for_prompt(base_prompt: str, beam_width: int, max_depth: int, target_length: int):
        node_counter = 0
        root_gen = generation_adapter(base_prompt, target_length)
        root_score = detector_adapter(root_gen)
        root_node = {
            "node_id": node_counter, "depth": 0, "parent_id": None, "applied_category": None,
            "constraint": None, "prompt_with_constraints": base_prompt, "generated_text": root_gen, "detector_score": root_score
        }
        node_counter += 1
        all_nodes = [root_node]
        beam = [root_node]
        categories = list(CATEGORY_DEFINITIONS.keys())

        for depth in range(1, max_depth + 1):
            children = []
            for parent in beam:
                parent_prompt = parent["prompt_with_constraints"]
                parent_text = parent["generated_text"]
                parent_score = lambda_score(parent)

                for cat in categories:
                    constraint_sentence = constraint_generator_adapter(
                        parent_prompt, parent_text, parent_score, cat, CATEGORY_DEFINITIONS[cat], target_length
                    )
                    new_prompt = append_constraint_to_prompt(parent_prompt, constraint_sentence)
                    gen_text = generation_adapter(new_prompt, target_length)
                    score = detector_adapter(gen_text)

                    node = {
                        "node_id": node_counter, "depth": depth, "parent_id": parent["node_id"],
                        "applied_category": cat, "constraint": constraint_sentence, "prompt_with_constraints": new_prompt,
                        "generated_text": gen_text, "detector_score": score
                    }
                    node_counter += 1
                    children.append(node)
                    all_nodes.append(node)

            if not children: break
            children_sorted = sorted(children, key=lambda x: lambda_score(x))
            beam = children_sorted[:beam_width]
            time.sleep(0.2)
        return all_nodes

    # --- MAIN ORCHESTRATION WITH CHECKPOINTING ---
    df = pd.read_csv(BASE_PROMPTS_CSV)
    print(f"Loaded {len(df)} base prompts.")

    # Checkpointing: Reload existing results if they exist to skip completed prompts
    completed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        if 'prompt_id' in existing_df.columns:
            completed_ids = set(existing_df['prompt_id'].unique())
        results = existing_df.to_dict('records')
        print(f"Found existing output file. Resuming. {len(completed_ids)} prompts already completed.")
    else:
        results = []

    for idx, row in df.iterrows():
        # Ensure we read the correct column name from the CSV
        base_prompt = str(row['prompt']) 
        prompt_id = row.get('id', idx)

        if prompt_id in completed_ids:
            print(f"Skipping prompt_id={prompt_id}, already processed.")
            continue

        print(f"\n=== Running beam search for prompt id={prompt_id} (row {idx}) ===")
        nodes = run_constraint_beam_search_for_prompt(
            base_prompt=base_prompt, beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH, target_length=TARGET_LENGTH
        )

        for n in nodes:
            n["prompt_row"] = int(idx)
            n["prompt_id"] = prompt_id
        
        results.extend(nodes)
        
        # Save intermediate results (Sync to Volume)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        vol.commit() # Critical: forces the volume to save state immediately
        print(f"Saved intermediate results to {OUTPUT_CSV}")

    print("\nAll prompts processed successfully.")

# Entry point to trigger the Modal deployment
@app.local_entrypoint()
def main():
    print("Submitting job to Modal...")
    run_experiment.remote()
    print("Job completed!")
