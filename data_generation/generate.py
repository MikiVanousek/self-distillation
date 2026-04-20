#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#


import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# =============================================================================
# Data Loading
# =============================================================================

def load_hf_dataset(dataset_name: str, config_name: str, split: str = "train") -> List[Dict[str, Any]]:
    """Load dataset from Hugging Face Hub."""
    print(f"Loading dataset '{dataset_name}' (config: {config_name}, split: {split})...")
    ds = load_dataset(dataset_name, config_name, split=split)
    examples = [dict(row) for row in ds]
    print(f"  Loaded {len(examples)} examples")
    return examples


# =============================================================================
# Helper Functions
# =============================================================================

def format_prompt(question, starter_code, problem_type, stdin_template, function_template):
    """Create prompt from templates."""
    if problem_type == 'function':
        return function_template.replace('{{ question }}', question or '').replace('{{ starter_code }}', starter_code or '')
    return stdin_template.replace('{{ question }}', question or '')


# =============================================================================
# Main Pipeline
# =============================================================================

def load_templates(template_dir: str) -> Tuple[str, str]:
    """Load prompt templates."""
    stdin_path = os.path.join(template_dir, "self_distillation_prompt_stdin.j2")
    function_path = os.path.join(template_dir, "self_distillation_prompt_function.j2")

    for path in [stdin_path, function_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template not found: {path}")

    with open(stdin_path) as f:
        stdin_template = f.read()
    with open(function_path) as f:
        function_template = f.read()

    return stdin_template, function_template


def _find_batch_size(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, Any]],
    max_new_tokens: int,
) -> int:
    """Binary-search for the largest batch size that fits in GPU memory.

    Runs a short dummy forward+generate pass at each candidate size,
    catches OOM, and backs off.  Returns at least 1.
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        return 1

    median_prompt_len = sorted(
        len(tokenizer.encode(ex["prompt"], add_special_tokens=False))
        for ex in examples
    )[len(examples) // 2]
    # Cap the trial generation length so probing is fast
    trial_gen_tokens = min(max_new_tokens, 32)

    dummy_ids = torch.full(
        (1, median_prompt_len), tokenizer.eos_token_id or 0,
        dtype=torch.long, device=device,
    )
    attn_mask = torch.ones_like(dummy_ids)

    lo, hi, best = 1, 128, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        ids = dummy_ids.expand(mid, -1).contiguous()
        mask = attn_mask.expand(mid, -1).contiguous()
        try:
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=trial_gen_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            best = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
        finally:
            torch.cuda.empty_cache()

    # Leave ~20% headroom for variable-length prompts
    batch_size = max(1, int(best * 0.8))
    print(f"  Auto batch size: {batch_size}  (probed max {best}, "
          f"median prompt {median_prompt_len} tokens)")
    return batch_size


def generate(config: Dict[str, Any], template_dir: str, limit: int = 0):
    """Run the data generation pipeline."""

    # Extract config
    model_name = config['model']['name']
    tp_size = config['model'].get('tensor_parallel_size', 4)
    gpu_mem = config['model'].get('gpu_memory_utilization', 0.85)

    dataset_name = config['dataset']['name']
    dataset_config = config['dataset']['config']
    dataset_split = config['dataset'].get('split', 'train')

    output_dir = config['output']['path']
    hf_repo = config['output'].get('hf_repo')

    temperature = config['generation'].get('temperature', 1.6)
    top_k = config['generation'].get('top_k', 20)
    top_p = config['generation'].get('top_p', 0.8)
    repetition_penalty = config['generation'].get('repetition_penalty', 1.0)
    max_tokens = config['generation'].get('max_tokens', 65536)

    filter_percent = config.get('post_process', {}).get('filter_shortest_percent', 10.0)

    if hf_repo:
        hf_token = os.environ.get("HF_TOKEN") or HfApi().token
        if not hf_token:
            print("Error: HF upload requested but no Hugging Face token found.\n"
                  "  Set HF_TOKEN env var or run `huggingface-cli login`.")
            sys.exit(1)
        print(f"HF upload: {hf_repo} (token found)")

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}/{dataset_config} (split: {dataset_split})")
    print(f"Output: {output_dir}")
    print(f"Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, "
          f"rep_penalty={repetition_penalty}, max_tokens={max_tokens}")
    print("-" * 60)

    # Load templates
    stdin_template, function_template = load_templates(template_dir)
    print("Templates loaded.")

    # ===== STAGE 0: Load and Format Data =====
    print("\n" + "=" * 60)
    print("STAGE 0: Load data + format prompts")
    stage0_start = time.time()

    examples = load_hf_dataset(dataset_name, dataset_config, split=dataset_split)
    if limit > 0:
        examples = examples[:limit]
        print(f"Limited to {len(examples)} examples")
    print(f"Total examples: {len(examples)}")

    for idx, ex in enumerate(examples):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Formatting: {idx}/{len(examples)} ({100*idx/len(examples):.1f}%)")

        # Infer problem_type from starter_code
        starter_code = ex.get('starter_code')
        ex['problem_type'] = 'function' if starter_code and starter_code.strip() else 'stdin'

        # Generate prompt
        ex['prompt'] = format_prompt(
            ex.get('question', ''), ex.get('starter_code'), ex['problem_type'],
            stdin_template, function_template,
        )

    print(f"STAGE 0 complete in {time.time() - stage0_start:.1f}s")

    # ===== STAGE 1: Generate Solutions =====
    print("\n" + "=" * 60)
    print("STAGE 1: Generate solutions")
    stage1_start = time.time()

    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    stop_token_ids = [
        tid for token in ["<|im_end|>", "<|endoftext|>"]
        if (tid := tokenizer.convert_tokens_to_ids(token)) is not None
        and tid != tokenizer.unk_token_id
    ]

    print(f"Generating solutions for {len(examples)} examples...")

    batch_size = _find_batch_size(model, tokenizer, examples, max_tokens)

    for batch_start in tqdm(range(0, len(examples), batch_size)):
        batch = examples[batch_start : batch_start + batch_size]
        prompts = [ex['prompt'] for ex in batch]

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=stop_token_ids or tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=gen_config)

        for i, ex in enumerate(batch):
            generated = output_ids[i][prompt_len:]
            ex['output'] = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"STAGE 1 complete: {len(examples)} generated in {time.time() - stage1_start:.1f}s")

    # ===== STAGE 2: Save Results =====
    print("\n" + "=" * 60)
    print("STAGE 2: Save results")

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "train.parquet")

    table = pa.Table.from_pylist(examples)
    pq.write_table(table, parquet_path)
    print(f"Saved {len(examples)} examples to {parquet_path}")

    # Print stats
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Total examples:    {len(examples)}")
    print(f"  Generated:         {len(examples)}")
    print("=" * 60)

    # ===== STAGE 3: Post-process to JSONL =====
    print("\n" + "=" * 60)
    print("STAGE 3: Post-process to training JSONL")
    stage3_start = time.time()

    # Collect valid responses
    valid_records = [
        ex for ex in examples
        if ex.get('output') and str(ex['output']).strip()
    ]

    # Length filtering
    min_length = 0
    if filter_percent > 0 and valid_records:
        lengths = sorted(len(str(ex['output']).strip()) for ex in valid_records)
        cutoff_idx = min(int(len(lengths) * filter_percent / 100.0), len(lengths) - 1)
        min_length = lengths[cutoff_idx]
        print(f"  Filter: dropping bottom {filter_percent}% shortest (min_length={min_length})")

    # Write JSONL
    jsonl_path = os.path.join(output_dir, "train.jsonl")

    kept = 0
    filtered = 0
    for ex in examples:
        prompt = ex.get('prompt', '')
        response = ex.get('output', '')

        if not prompt or not str(prompt).strip():
            filtered += 1
            continue
        if not response or not str(response).strip():
            filtered += 1
            continue

        prompt = str(prompt).strip()
        response = str(response).strip()

        if len(response) < min_length:
            filtered += 1
            continue

        entry = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
        with open(jsonl_path, "a" if kept > 0 else "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        kept += 1

    print(f"STAGE 3 complete in {time.time() - stage3_start:.1f}s")
    print(f"  Total records:  {len(examples)}")
    print(f"  Kept:           {kept} ({100*kept/len(examples):.1f}%)")
    print(f"  Filtered:       {filtered}")
    print(f"  Output:         {jsonl_path}")

    # ===== STAGE 4: Upload to Hugging Face Hub =====
    if hf_repo:
        print("\n" + "=" * 60)
        print(f"STAGE 4: Upload to Hugging Face Hub ({hf_repo})")
        stage4_start = time.time()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        conversations = [record["messages"] for record in records]
        hf_dataset = Dataset.from_dict({"messages": conversations})

        hf_dataset.push_to_hub(hf_repo, private=False)

        print(f"STAGE 4 complete in {time.time() - stage4_start:.1f}s")
        print(f"  Uploaded {len(hf_dataset)} conversations to https://huggingface.co/datasets/{hf_repo}")
    else:
        print("\nSkipping HF upload (no hf_repo configured)")


def main():
    parser = argparse.ArgumentParser(description="Generate solutions for coding problems using vLLM")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--temperature", type=float, help="Override generation temperature")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--dataset-name", type=str, help="Override HuggingFace dataset name")
    parser.add_argument("--output-path", type=str, help="Override output path")
    parser.add_argument("--hf-repo", type=str, help="Override HF repo to upload dataset to")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 = all)")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.temperature is not None:
        config['generation']['temperature'] = args.temperature
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.dataset_name:
        config['dataset']['name'] = args.dataset_name
    if args.output_path:
        config['output']['path'] = args.output_path
    if args.hf_repo:
        config['output']['hf_repo'] = args.hf_repo

    # Resolve template directory (templates/ next to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, "templates")

    generate(config, template_dir, limit=args.limit)


if __name__ == "__main__":
    main()
