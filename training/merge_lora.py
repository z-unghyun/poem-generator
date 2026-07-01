"""Merge a LoRA adapter into Qwen/Qwen2.5-0.5B-Instruct and optionally upload it.

This script is intended to run after Colab LoRA training.
It saves a merged model for simpler Hugging Face Space CPU deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", required=True, help="Local adapter path or Hugging Face adapter repo ID")
    parser.add_argument("--output_dir", default="outputs/merged-model")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None, help="Target Hugging Face Hub repo for the merged model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set")
        merged_model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print(f"Uploaded merged model to {args.hub_model_id}")


if __name__ == "__main__":
    main()
