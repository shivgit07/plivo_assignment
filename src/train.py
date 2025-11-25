import os
import argparse
import torch
import sys
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure src modules can be imported
sys.path.append(os.path.join(os.getcwd(), 'src'))

from dataset import PIIDataset, collate_batch
from labels import LABELS, LABEL2ID
from model import create_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilroberta-base")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl") 
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10) 
    ap.add_argument("--lr", type=float, default=3e-5) # Slightly lower LR for RoBERTa
    ap.add_argument("--max_length", type=int, default=128) 
    ap.add_argument("--freeze_layers", type=int, default=3) 
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def calculate_class_weights(dataset, device):
    """Calculates inverse frequency weights to penalize missing rare entities."""
    print("Calculating class weights...")
    label_counts = Counter()
    for item in dataset:
        # filter out -100 (ignore index)
        valid_labels = [l for l in item['labels'] if l != -100]
        label_counts.update(valid_labels)
    
    total_count = sum(label_counts.values())
    num_classes = len(LABELS)
    
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total_count / (num_classes * count))
            
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    weights = torch.clamp(weights, min=1.0, max=10.0) 
    print(f"Class Weights applied: {weights}")
    return weights

def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1, len(LABELS))
            labels_flat = labels.view(-1)
            loss = loss_fn(logits, labels_flat)
            
            total_loss += loss.item()
    
    return total_loss / max(1, len(dataloader))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    print(f"Loading training data from {args.train}...")
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    print(f"Loading dev data from {args.dev}...")
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_dl = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)

    # --- FIX: DYNAMIC FREEZING LOGIC ---
    if args.freeze_layers > 0:
        print(f"Freezing embeddings and first {args.freeze_layers} layers...")
        
        # Detect model type to access the correct base model
        if hasattr(model, "distilbert"):
            base_model = model.distilbert
            print("Detected DistilBERT architecture.")
        elif hasattr(model, "roberta"):
            base_model = model.roberta
            print("Detected RoBERTa architecture.")
        elif hasattr(model, "bert"):
            base_model = model.bert
            print("Detected BERT architecture.")
        else:
            print("Warning: Could not detect base model type. Skipping freeze.")
            base_model = None

        if base_model:
            # Freeze Embeddings
            for param in base_model.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze Bottom Layers
            # Note: RoBERTa/BERT use 'encoder.layer', DistilBERT uses 'transformer.layer'
            if hasattr(base_model, "encoder"):
                layers = base_model.encoder.layer
            elif hasattr(base_model, "transformer"):
                layers = base_model.transformer.layer
            else:
                layers = []

            for i in range(args.freeze_layers):
                if i < len(layers):
                    for param in layers[i].parameters():
                        param.requires_grad = False
    # -----------------------------------

    class_weights = calculate_class_weights(train_ds.items, args.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    print("Starting training...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits.view(-1, len(LABELS))
            labels_flat = labels.view(-1)
            loss = loss_fn(logits, labels_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, len(train_dl))
        avg_val_loss = evaluate(model, dev_dl, args.device, loss_fn)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"  -> New best model saved (Val Loss: {avg_val_loss:.4f})")

    print("Training complete.")

if __name__ == "__main__":
    main()