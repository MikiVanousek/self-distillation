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
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

import sglang as sgl


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


def generate(config: Dict[str, Any], template_dir: str, limit: int = 0):
    """Run the data generation pipeline."""

    # Extract config
    model_name = config['model']['name']
    tp_size = config['model'].get('tensor_parallel_size', 1)
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
    print(f"Engine: SGLang (tp={tp_size}, gpu_mem={gpu_mem})")
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

    # Load tokenizer for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for idx, ex in enumerate(examples):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Formatting: {idx}/{len(examples)} ({100*idx/len(examples):.1f}%)")

        starter_code = ex.get('starter_code')
        ex['problem_type'] = 'function' if starter_code and starter_code.strip() else 'stdin'

        ex['prompt'] = format_prompt(
            ex.get('question', ''), ex.get('starter_code'), ex['problem_type'],
            stdin_template, function_template,
        )

        # Apply chat template so SGLang receives the full formatted text
        messages = [{"role": "user", "content": ex['prompt']}]
        ex['formatted_prompt'] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    print(f"STAGE 0 complete in {time.time() - stage0_start:.1f}s")

    # ===== STAGE 1: Generate Solutions =====
    print("\n" + "=" * 60)
    print("STAGE 1: Generate solutions")
    stage1_start = time.time()

    print(f"Loading SGLang engine: {model_name} ...")
    llm = sgl.Engine(
        model_path=model_name,
        tp_size=tp_size,
        mem_fraction_static=gpu_mem,
        trust_remote_code=True,
    )

    sampling_params = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_tokens,
    }

    prompts = [ex['formatted_prompt'] for ex in examples]
    print(f"Generating solutions for {len(prompts)} prompts...")

    outputs = llm.generate(prompts, sampling_params)

    for ex, out in zip(examples, outputs):
        ex['output'] = out['text'].strip()

    llm.shutdown()

    print(f"STAGE 1 complete: {len(examples)} generated in {time.time() - stage1_start:.1f}s")

    # ===== STAGE 2: Save Results =====
    print("\n" + "=" * 60)
    print("STAGE 2: Save results")

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "train.parquet")

    # Drop formatted_prompt before saving (internal field)
    save_examples = [{k: v for k, v in ex.items() if k != 'formatted_prompt'} for ex in examples]
    table = pa.Table.from_pylist(save_examples)
    pq.write_table(table, parquet_path)
    print(f"Saved {len(examples)} examples to {parquet_path}")

    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Total examples:    {len(examples)}")
    print(f"  Generated:         {len(examples)}")
    print("=" * 60)

    # ===== STAGE 3: Post-process to JSONL =====
    print("\n" + "=" * 60)
    print("STAGE 3: Post-process to training JSONL")
    stage3_start = time.time()

    valid_records = [
        ex for ex in examples
        if ex.get('output') and str(ex['output']).strip()
    ]

    min_length = 0
    if filter_percent > 0 and valid_records:
        lengths = sorted(len(str(ex['output']).strip()) for ex in valid_records)
        cutoff_idx = min(int(len(lengths) * filter_percent / 100.0), len(lengths) - 1)
        min_length = lengths[cutoff_idx]
        print(f"  Filter: dropping bottom {filter_percent}% shortest (min_length={min_length})")

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
    parser = argparse.ArgumentParser(description="Generate solutions using SGLang offline engine")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--temperature", type=float, help="Override generation temperature")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--dataset-name", type=str, help="Override HuggingFace dataset name")
    parser.add_argument("--output-path", type=str, help="Override output path")
    parser.add_argument("--hf-repo", type=str, help="Override HF repo to upload dataset to")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 = all)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, "templates")

    generate(config, template_dir, limit=args.limit)


if __name__ == "__main__":
    main()
