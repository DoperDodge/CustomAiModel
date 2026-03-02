"""
Reasoning Fine-Tuning — Train on math & reasoning datasets with chain-of-thought

Fine-tunes the base LLM (Phi-3) using LoRA on reasoning-focused datasets
to improve step-by-step problem solving, math, and logical reasoning.

Supported datasets:
    - openai/gsm8k          (8.5K grade-school math problems with step-by-step solutions)
    - Open-Orca/OpenOrca    (4M instruction-following, sampled down for training)
    - meta-math/MetaMathQA  (395K math problems with detailed solutions)

Usage:
    # Train on GSM8K only
    python -m src.text_to_text.train_reasoning --datasets gsm8k

    # Train on a mix of all three (recommended)
    python -m src.text_to_text.train_reasoning --datasets gsm8k openorca metamath

    # Custom sample sizes
    python -m src.text_to_text.train_reasoning \
        --datasets gsm8k openorca metamath \
        --max_samples_per_dataset 5000

    # Use a specific base model
    python -m src.text_to_text.train_reasoning \
        --base_model microsoft/Phi-3-mini-4k-instruct \
        --datasets gsm8k metamath
"""

import argparse
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


# ──────────────────────────────────────────────
# Reasoning system prompt — encourages chain-of-thought
# ──────────────────────────────────────────────

REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step. "
    "Think through each problem carefully, show your reasoning, "
    "and provide a clear final answer."
)

# ──────────────────────────────────────────────
# Dataset registry — maps names to loader functions
# ──────────────────────────────────────────────

DATASET_REGISTRY: dict[str, dict] = {
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "hf_config": "main",
        "split": "train",
        "description": "Grade-school math (8.5K examples)",
    },
    "openorca": {
        "hf_id": "Open-Orca/OpenOrca",
        "hf_config": None,
        "split": "train",
        "description": "Instruction-following mix (4M examples, sampled)",
        "default_max_samples": 10000,
    },
    "metamath": {
        "hf_id": "meta-math/MetaMathQA",
        "hf_config": None,
        "split": "train",
        "description": "Math with detailed solutions (395K examples)",
        "default_max_samples": 10000,
    },
}


def list_available_datasets() -> list[str]:
    """Return names of all supported reasoning datasets."""
    return list(DATASET_REGISTRY.keys())


# ──────────────────────────────────────────────
# Formatters — convert each dataset's schema to chat messages
# ──────────────────────────────────────────────

def format_gsm8k(example: dict) -> list[dict]:
    """Format a GSM8K example into chat messages.

    GSM8K schema: {"question": str, "answer": str}
    The answer field contains step-by-step work followed by #### <final_answer>.
    """
    question = example["question"]
    raw_answer = example["answer"]

    # Split on #### to separate reasoning from final answer
    if "####" in raw_answer:
        reasoning, final = raw_answer.rsplit("####", 1)
        reasoning = reasoning.strip()
        final = final.strip()
        answer = f"{reasoning}\n\n**Answer: {final}**"
    else:
        answer = raw_answer

    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def format_openorca(example: dict) -> list[dict]:
    """Format an OpenOrca example into chat messages.

    OpenOrca schema: {"system_prompt": str, "question": str, "response": str}
    """
    messages = []
    system = example.get("system_prompt", "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    else:
        messages.append({"role": "system", "content": REASONING_SYSTEM_PROMPT})

    messages.append({"role": "user", "content": example["question"]})
    messages.append({"role": "assistant", "content": example["response"]})
    return messages


def format_metamath(example: dict) -> list[dict]:
    """Format a MetaMathQA example into chat messages.

    MetaMathQA schema: {"query": str, "response": str, "type": str}
    """
    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]


FORMATTERS = {
    "gsm8k": format_gsm8k,
    "openorca": format_openorca,
    "metamath": format_metamath,
}


# ──────────────────────────────────────────────
# Dataset loading and mixing
# ──────────────────────────────────────────────

def load_reasoning_dataset(
    name: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load a single reasoning dataset from HuggingFace.

    Args:
        name: Dataset name (must be in DATASET_REGISTRY).
        max_samples: Max examples to use. None = use all.
        seed: Random seed for sampling.

    Returns:
        HuggingFace Dataset with a "_source" column added.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[name]
    hf_id = info["hf_id"]
    hf_config = info.get("hf_config")
    split = info["split"]

    print(f"[Reasoning] Loading {name} ({hf_id})...")

    load_kwargs = {"split": split}
    if hf_config:
        load_kwargs["name"] = hf_config

    dataset = load_dataset(hf_id, **load_kwargs)

    # Apply default max_samples if dataset is very large
    effective_max = max_samples or info.get("default_max_samples")
    if effective_max and len(dataset) > effective_max:
        print(f"[Reasoning] Sampling {effective_max} from {len(dataset)} examples")
        dataset = dataset.shuffle(seed=seed).select(range(effective_max))

    # Tag with source for tracking
    dataset = dataset.map(lambda x: {"_source": name})

    print(f"[Reasoning] {name}: {len(dataset)} examples loaded")
    return dataset


def mix_datasets(
    names: list[str],
    max_samples_per_dataset: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load and concatenate multiple reasoning datasets.

    Args:
        names: List of dataset names to mix.
        max_samples_per_dataset: Max samples from each dataset.
        seed: Random seed for sampling and shuffling.

    Returns:
        Concatenated and shuffled Dataset.
    """
    datasets = []
    for name in names:
        ds = load_reasoning_dataset(name, max_samples=max_samples_per_dataset, seed=seed)
        datasets.append(ds)

    if len(datasets) == 1:
        combined = datasets[0]
    else:
        # Align columns — keep only columns present in all datasets plus _source
        common_cols = set(datasets[0].column_names)
        for ds in datasets[1:]:
            common_cols &= set(ds.column_names)
        common_cols.add("_source")

        aligned = []
        for ds in datasets:
            drop_cols = [c for c in ds.column_names if c not in common_cols]
            if drop_cols:
                ds = ds.remove_columns(drop_cols)
            aligned.append(ds)

        combined = concatenate_datasets(aligned)

    combined = combined.shuffle(seed=seed)
    print(f"[Reasoning] Combined dataset: {len(combined)} examples")
    return combined


def format_example_for_training(example: dict, tokenizer) -> str:
    """Format a dataset example into a training string using the appropriate formatter.

    Args:
        example: A single dataset example with a "_source" field.
        tokenizer: The tokenizer to apply chat template.

    Returns:
        Formatted string ready for training.
    """
    source = example.get("_source", "")
    formatter = FORMATTERS.get(source)

    if formatter is None:
        # Fallback: try to infer format
        if "question" in example and "answer" in example:
            formatter = format_gsm8k
        elif "query" in example and "response" in example:
            formatter = format_metamath
        elif "question" in example and "response" in example:
            formatter = format_openorca
        else:
            raise ValueError(f"Cannot format example with keys: {list(example.keys())}")

    messages = formatter(example)
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Run reasoning fine-tuning."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization for memory efficiency
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
    }

    if args.quantize:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
            print("[Reasoning] Using QLoRA (4-bit quantization)")
        except ImportError:
            print("[Reasoning] bitsandbytes not available, using full precision")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and mix datasets
    dataset = mix_datasets(
        names=args.datasets,
        max_samples_per_dataset=args.max_samples_per_dataset,
        seed=args.seed,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb" if args.wandb else "none",
        gradient_checkpointing=True,
        optim="adamw_8bit",
        max_length=args.max_seq_len,
        seed=args.seed,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=lambda example: format_example_for_training(example, tokenizer),
    )

    # Train
    print(f"Starting reasoning fine-tuning ({len(dataset)} examples, {args.epochs} epochs)...")
    trainer.train()

    # Save
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    import json
    meta = {
        "base_model": args.base_model,
        "datasets": args.datasets,
        "total_examples": len(dataset),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "quantized": args.quantize,
    }
    meta_path = Path(args.output_dir) / "reasoning_training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Training metadata saved to {meta_path}")
    print("Reasoning fine-tuning complete.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on reasoning datasets")
    parser.add_argument(
        "--base_model", type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["gsm8k", "metamath"],
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to train on (space-separated)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./checkpoints/t2t-reasoning",
        help="Directory to save the fine-tuned LoRA weights",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--max_samples_per_dataset", type=int, default=None,
        help="Max samples from each dataset (None = use dataset default)",
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantize", action="store_true", default=True,
        help="Use QLoRA (4-bit quantization) — recommended for 8GB GPUs",
    )
    parser.add_argument("--no_quantize", action="store_false", dest="quantize")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
