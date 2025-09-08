#!/usr/bin/env python3
import os
import math
import csv
import argparse
import random
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)

# ---------- Device helpers ----------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------- Data loading ----------
def load_txt_folder(folder: str) -> List[str]:
    texts = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
    if not texts:
        raise ValueError(f"No non-empty .txt files found in {folder!r}")
    return texts

# ---------- CSV logger ----------
class CsvLossLogger(TrainerCallback):
    """
    Logs {"step","epoch","learning_rate","train_loss"} on log steps
    and {"eval_loss","perplexity"} on evaluations to a CSV.
    Keeps the file open across post-training evaluate() calls.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._file = None
        self._writer = None

    def _ensure_open(self, append=False):
        mode = "a" if append else "w"
        if self._file is None or self._file.closed:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self._file = open(self.csv_path, mode, newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if mode == "w":
                self._writer.writerow(
                    ["type", "step", "epoch", "learning_rate", "train_loss", "eval_loss", "perplexity"]
                )
                self._file.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        # Always (re)create the file at the start of training
        self._ensure_open(append=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        self._ensure_open(append=True)
        step = int(state.global_step) if state.global_step is not None else None
        epoch = float(state.epoch) if state.epoch is not None else None
        lr = logs.get("learning_rate")
        train_loss = logs.get("loss")
        if train_loss is not None:
            self._writer.writerow(["train", step, epoch, lr, train_loss, None, None])
            self._file.flush()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self._ensure_open(append=True)
        step = int(state.global_step) if state.global_step is not None else None
        epoch = float(state.epoch) if state.epoch is not None else None
        eval_loss = metrics.get("eval_loss")
        ppl = math.exp(eval_loss) if (eval_loss is not None and eval_loss < 20) else None
        self._writer.writerow(["eval", step, epoch, None, None, eval_loss, ppl])
        self._file.flush()


# ---------- Tokenization ----------
def make_tokenize_fn(tokenizer, max_length: int):
    def _tok(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    return _tok

# ---------- Generation ----------
def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt")
    # move inputs to model.device (CPU/MPS/CUDA safe)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune distilgpt2 on Poe poems stored as .txt files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing .txt poems.")
    parser.add_argument("--output_dir", type=str, default="poe-distilgpt2-finetuned", help="Checkpoint dir.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 even on CUDA.")
    parser.add_argument("--prompt", type=str, default="Upon the midnight shore I stood, and thought of her—")
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    poems = load_txt_folder(args.data_dir)
    random.shuffle(poems)

    ds = Dataset.from_dict({"text": poems})
    if args. eval_ratio and args.eval_ratio > 0:
        n_eval = max(1, int(len(ds) * args.eval_ratio)) if len(ds) > 4 else 1
        ds = ds.train_test_split(test_size=n_eval, seed=args.seed)
        train_ds = ds["train"]
        eval_ds = ds["test"]
        has_eval = True
    else:
        train_ds = ds
        eval_ds = None
        has_eval = False

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Device
    device = get_device()
    model.to(device)

    # Tokenize
    tok_fn = make_tokenize_fn(tokenizer, args.max_length)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=["text"]) if has_eval else None

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Precision guards (disable half-precision on MPS)
    fp16_ok = torch.cuda.is_available() and (not args.no_fp16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        eval_strategy="epoch" if has_eval and args.epochs > 0 else "no",
        report_to="none",
        fp16=False if not fp16_ok else True,  # never fp16 on CPU/MPS
        bf16=False,                            # keep off for MPS stability
    )

    csv_logger = CsvLossLogger(csv_path=os.path.join(args.output_dir, "training_log.csv"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[csv_logger],
    )

    # Train (can be 0 epochs to skip)
    if args.epochs > 0:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        # Load previously saved (if exists); otherwise keep current weights
        try:
            model = AutoModelForCausalLM.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model.to(get_device())
        except Exception:
            pass

    # Optional final eval
    if has_eval:
        metrics = trainer.evaluate(eval_dataset=eval_tok)
        print("\nValidation metrics:", metrics)
        if "eval_loss" in metrics:
            try:
                print("Validation perplexity:", math.exp(metrics["eval_loss"]))
            except OverflowError:
                print("Validation perplexity: overflow")

    # Sample generation
    print("\n--- Sample Generation ---")
    sample = generate_sample(model, tokenizer, prompt=args.prompt, max_new_tokens=80)
    print(sample)
    print("-------------------------")

    print(f"\nSaved/using model at: {args.output_dir}")
    print(f"Training CSV (if trained): {os.path.join(args.output_dir, 'training_log.csv')}")

if __name__ == "__main__":
    main()
