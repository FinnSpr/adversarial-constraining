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
app = modal.App("ai-safety-coherence-check")

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
    EXPERIMENT_RESULTS_CSV = "/mnt/ai-safety/constraint_search_results_full.csv"
    OUTPUT_CSV = "/mnt/ai-safety/coherence_scores.csv"

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
        generated_ids = qwen_model.generate(**model_inputs, max_new_tokens=max_new_tokens, pad_token_id=qwen_tokenizer.eos_token_id, do_sample=False, temperature=0)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = qwen_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content

    # --- PROMPT TEMPLATES ---
    coherence_prompt_template = (
        "The following prompt was given to an LLM:\n"
        "{FULL_PROMPT}\n\n"
        "The LLM produced the following output:\n"
        "{LLM_OUTPUT}\n\n"
        "Task: Rate how coherent the output is on a scale from 1 to 5 (1 = incoherent, 5 = perfectly coherent).\n"
        "Guidelines: Judge coherence relative to the original prompt and its constraints. Consider logical flow, clarity of ideas, sentence-to-sentence connections, and whether the output stays on task. Penalize needless repetition, circular phrasing, or repeated fragments that do not advance meaning.\n"
        "Output requirement: Return exactly one integer (1,2,3,4,or 5) and nothing else.\n"
        "Your rating:"
    )


    def coherence_scoring(node_id, full_prompt, llm_output):
        filled = coherence_prompt_template.format(FULL_PROMPT=full_prompt, LLM_OUTPUT=llm_output)
        gen = qwen_generate(filled, enable_thinking=False, max_new_tokens=4)
        gen = gen.strip()
        m = re.search(r"[1-5]", gen)
        if m:
            score = int(m.group(0))
        else:
            score = 1
            print(f"Coherence scoring failed for node ID {node_id}, LLM judge output was {gen}, setting score to 1.")
        return score

    # --- MAIN ORCHESTRATION WITH CHECKPOINTING ---
    df = pd.read_csv(EXPERIMENT_RESULTS_CSV)
    assert len(df) == 50*(1+6+18+18+18+18)
    print(f"Loaded {len(df)} experiment results.")

    # Checkpointing: Reload existing results if they exist to skip completed prompts
    completed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        if 'node_id' in existing_df.columns:
            completed_ids = set(existing_df['node_id'].unique())
        coherence_results = existing_df.to_dict('records')
        print(f"Found existing output file. Resuming. {len(completed_ids)} judgments already completed.")
    else:
        coherence_results = []

    for idx, row in df.iterrows():
        # Ensure we read the correct column name from the CSV
        node_id = str(row['node_id'])
        prompt = row["prompt_with_constraints"]
        output = row["generated_text"]

        if node_id in completed_ids:
            continue
        
        score = coherence_scoring(node_id, prompt, output)
        coherence_score = {
            "node_id": node_id, "coherence_score": score
        }
        coherence_results.append(coherence_score)
        
        if idx % 50 == 0:           
            # Save intermediate results (Sync to Volume)
            pd.DataFrame(coherence_results).to_csv(OUTPUT_CSV, index=False)
            vol.commit() # Critical: forces the volume to save state immediately
            print(f"Saved intermediate results to {OUTPUT_CSV}, current index: {idx}")

    pd.DataFrame(coherence_results).to_csv(OUTPUT_CSV, index=False)
    vol.commit() # Critical: forces the volume to save state immediately
    print(f"Saved final results to {OUTPUT_CSV}")
    print("\nAll results processed successfully.")

# Entry point to trigger the Modal deployment
@app.local_entrypoint()
def main():
    print("Submitting job to Modal...")
    run_experiment.remote()
    print("Job completed!")
