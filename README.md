## Generate Poe style verses console application

### Features
Fine-tune a small open-source language mode
(distilgpt2) on a tiny, public-domain dataset of Edgar Allan Poe poems,
then generate Poe-style verse.
Works on CPU, CUDA GPUs, and Apple Silicon (MPS), with careful device handling.

### Files
#### poe_finetune.py
- All-in-one script with a built-in list of Poe poems (POE_POEMS).
- Does a train/val/test split (60/20/20).
- Fine-tunes distilgpt2, evaluates on the test set (loss + perplexity), generates a sample poem, and saves the model.

#### train_poe_distilgpt2.py
- File-based trainer that loads each poem from a folder of .txt files.
- CLI flags for epochs, batch size, LR, max_length, etc.
- Validation split via --eval_ratio.
- Logs per-step train loss and per-epoch eval loss + perplexity to training_log.csv.
- Generates a sample poem at the end.
- Also CPU/MPS/CUDA-safe; fp16 disabled on non-CUDA.

### Simple
##### Will train on first use, after first run uses the local saved model
- python poe_finetune.py

### Retrain
- python poe_finetune.py --mode train

### Only generate
- python poe_finetune.py --mode generate --prompt "Once upon a midnight dreary,"

### Different output
- python poe_finetune.py --output_dir poe-v2



### Controlled
##### Train
- python train_poe_distilgpt2.py \
  --data_dir poe_dataset \
  --output_dir poe-distilgpt2-finetuned \
  --epochs 8 \
  --eval_ratio 0.2


##### Generate example
- python train_poe_distilgpt2.py \
  --data_dir poe_dataset \
  --output_dir poe-distilgpt2-finetuned \
  --epochs 0 \
  --prompt "In a kingdom by the sea,"
