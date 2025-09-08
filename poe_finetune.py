#!/usr/bin/env python3
import os
import math
import random
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------- Device helpers ----------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def checkpoint_exists(path: str) -> bool:
    needed = ["config.json", "tokenizer_config.json"]
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, f)) for f in needed)

# ---------- Poe dataset (public domain) ----------
POE_POEMS = [
    # A Dream Within a Dream
    """Take this kiss upon the brow!
And, in parting from you now,
Thus much let me avow—
You are not wrong, who deem
That my days have been a dream;
Yet if hope has flown away
In a night, or in a day,
In a vision, or in none,
Is it therefore the less gone?
All that we see or seem
Is but a dream within a dream.

I stand amid the roar
Of a surf-tormented shore,
And I hold within my hand
Grains of the golden sand—
How few! yet how they creep
Through my fingers to the deep,
While I weep—while I weep!
O God! can I not grasp
Them with a tighter clasp?
O God! can I not save
One from the pitiless wave?
Is all that we see or seem
But a dream within a dream?""",

    # Annabel Lee (truncated here just to keep script short)
    """It was many and many a year ago,
In a kingdom by the sea,
That a maiden there lived whom you may know
By the name of Annabel Lee—
...
In the sepulchre there by the sea—
In her tomb by the sounding sea.""",

    # Eldorado
    """Gaily bedight,
A gallant knight,
In sunshine and in shadow,
Had journeyed long,
Singing a song,
In search of Eldorado.
...
“If you seek for Eldorado!”""",

    # Alone
    """From childhood’s hour I have not been
As others were—I have not seen
As others saw—I could not bring
My passions from a common spring.
...
Of a demon in my view.""",

    # To Helen
    """Helen, thy beauty is to me
Like those Nicean barks of yore,
That gently, o’er a perfumed sea,
The weary, way-worn wanderer bore
To his own native shore.""",
]

# ---------- Helpers ----------
def tokenize_fn(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 80):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune or reuse a distilgpt2 Poe model.")
    parser.add_argument("--output_dir", type=str, default="poe-finetuned",
                        help="Where to save/load the model.")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "train", "generate", "eval"],
                        help="auto: train if no checkpoint, else reuse; train: always train; generate: load & generate; eval: load & test-eval")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="Upon the midnight shore I stood, and thought of her—")
    args = parser.parse_args()

    set_seed(args.seed)

    # Decide train vs reuse
    have_ckpt = checkpoint_exists(args.output_dir)
    will_train = (args.mode == "train") or (args.mode == "auto" and not have_ckpt)

    # Build dataset & split (only actually needed for train/eval)
    random.shuffle(POE_POEMS)
    ds = Dataset.from_dict({"text": POE_POEMS})
    ds = ds.train_test_split(test_size=0.4, seed=args.seed)
    temp = ds["test"].train_test_split(test_size=0.5, seed=args.seed)
    train_ds, val_ds, test_ds = ds["train"], temp["train"], temp["test"]

    # Tokenizer
    base_name = "distilgpt2"
    tok_src = args.output_dir if have_ckpt else base_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model_src = args.output_dir if have_ckpt and not will_train else base_name
    model = AutoModelForCausalLM.from_pretrained(model_src)

    # Device
    device = get_device()
    model.to(device)

    # Tokenize
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tok = lambda b: tokenize_fn(b, tokenizer, args.max_length)
    train_tok = train_ds.map(tok, batched=True, remove_columns=["text"])
    val_tok   = val_ds.map(tok, batched=True, remove_columns=["text"])
    test_tok  = test_ds.map(tok, batched=True, remove_columns=["text"])

    # Choose eval strategy based on mode
    eval_strategy = "epoch" if (will_train or args.mode in ["eval", "auto"]) else "no"

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=2,
        eval_strategy=eval_strategy,   # <-- dynamic
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok if will_train else None,
        # only provide eval datasets if we plan to evaluate
        eval_dataset=val_tok if eval_strategy != "no" else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # TRAIN (if needed)
    if will_train:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        have_ckpt = True

    # EVAL (only in eval/auto modes)
    if eval_strategy != "no":
        metrics = trainer.evaluate(eval_dataset=test_tok)
        print("\n--- Test Set Metrics ---")
        print(metrics)
        if "eval_loss" in metrics:
            try:
                print(f"Test perplexity: {math.exp(metrics['eval_loss']):.2f}")
            except OverflowError:
                print("Test perplexity: overflow")

    # GENERATE (always print a sample)
    print("\n--- Sample Generation ---")
    print(generate_sample(model, tokenizer, args.prompt))
    print("-------------------------")

    print(f"\nUsing model at: {args.output_dir}")
    if will_train:
        print("Trained this run (and saved).")
    else:
        print("Reused existing checkpoint (no training).")

if __name__ == "__main__":
    main()
