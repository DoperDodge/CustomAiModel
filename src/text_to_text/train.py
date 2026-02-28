"""
Text-to-Text Training — Fine-tuning with LoRA

This script fine-tunes a pre-trained LLM using LoRA (Low-Rank Adaptation)
for chat / instruction-following tasks.

Usage:
    python -m src.text_to_text.train \
        --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --dataset OpenAssistant/oasst2 \
        --output_dir ./checkpoints/t2t-chat \
        --epochs 3
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


def format_chat_example(example: dict, tokenizer) -> str:
    """Format a dataset example into a chat prompt.

    Adjust this function to match your dataset's schema.
    """
    messages = []
    if "instruction" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content += "\n\n" + example["input"]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": example["output"]})
    elif "messages" in example:
        messages = example["messages"]
    elif "prompt" in example and "response" in example:
        messages.append({"role": "user", "content": example["prompt"]})
        messages.append({"role": "assistant", "content": example["response"]})
    else:
        raise ValueError(f"Unknown dataset format. Keys: {list(example.keys())}")

    return tokenizer.apply_chat_template(messages, tokenize=False)


def train(args: argparse.Namespace) -> None:
    # Load base model
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb" if args.wandb else "none",
        gradient_checkpointing=True,
        optim="adamw_8bit",
        max_seq_length=args.max_seq_len,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=lambda example: format_chat_example(example, tokenizer),
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/t2t-chat")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
