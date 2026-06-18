#!/usr/bin/env python3
"""
SYSMH-S-Triage training script.
Emergency triage at SYSU Sun Yat-sen Memorial Hospital South Campus; 4-class task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class SYSMHSTriageDataset(Dataset):
    """SYSMH-S-Triage dataset."""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read CSV data.
        self.data = pd.read_csv(data_path)
        print(f"Loaded {len(self.data)} records")
        
        # Label mapping: 1,2,3,4 -> 0,1,2,3.
        self.data['label'] = self.data['label'] - 1
        
        # Class distribution.
        label_counts = self.data['label'].value_counts().sort_index()
        print(f"Class distribution: {dict(label_counts)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, train_loader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics.
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def evaluate(model, val_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='SYSMH-S-Triage Training')
    
    parser.add_argument('--model_path', type=str, default='../../pytorch_model.bin',
                       help='Pretrained model path')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Training data path (CSV format)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Validation data path (CSV format)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/sysmh_s_triage',
                       help='Save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    
    args = parser.parse_args()
    
    # Create save directory.
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("SYSMH-S-Triage training - 4-class ED triage")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.val_data}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Load model.
    print("[1/4] Loading model...")
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path,
        dataset_name='SYSMH-S-Triage',
        task_type='early_triage'
    )
    model.freeze_encoders()  # Ensure encoders are frozen.
    model = model.to(args.device)
    print("Model loaded\n")
    
    # Load data.
    print("[2/4] Loading data...")
    tokenizer = SimpleEDTokenizer()
    train_dataset = SYSMHSTriageDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = SYSMHSTriageDataset(args.val_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Data loaded\n")
    
    # Set optimizer.
    print("[3/4] Setting optimizer...")
    optimizer = optim.AdamW(
        model.task_head.parameters(),  # Optimize only the task head.
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    print("Optimizer ready\n")
    
    # Train.
    print("[4/4] Starting training...")
    print("-"*80)
    
    best_f1 = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-"*80)
        
        # Train.
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, args.device)
        
        # Validate.
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, args.device)
        
        # Update learning rate.
        scheduler.step()
        
        # Record history.
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
        
        # Print results.
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model.
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save(checkpoint, save_path)
            print(f"  Saved best model (F1: {val_f1:.4f})")
            
            # Save classification report.
            report = classification_report(
                val_labels,
                val_preds,
                target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                digits=4
            )
            with open(os.path.join(args.save_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
    
    # Save training history.
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final report.
    print("\n" + "="*80)
    print("Training complete.")
    print("="*80)
    print(f"Best model: Epoch {best_epoch}, F1: {best_f1:.4f}")
    print(f"Model saved in: {args.save_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
