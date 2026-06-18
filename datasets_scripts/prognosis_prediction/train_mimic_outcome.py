#!/usr/bin/env python3
"""
MIMIC-IV-ED-Outcome training script.
Multimodal text-and-ECG multi-task learning.
Tasks: 1) admission prediction, 2) length of stay, 3) severity scoring.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class MIMICOutcomeDataset(Dataset):
    """MIMIC-IV-ED-Outcome multimodal multi-task dataset."""
    def __init__(self, data_path, ecg_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ecg_dir = ecg_dir
        self.data = pd.read_csv(data_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.data)} multimodal records")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Text.
        encoding = self.tokenizer(str(row['text']), max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
        
        # ECG image.
        ecg_path = os.path.join(self.ecg_dir, row['ecg_path'])
        try:
            ecg_image = Image.open(ecg_path).convert('RGB')
            pixel_values = self.transform(ecg_image)
        except:
            pixel_values = torch.zeros(3, 224, 224)  # Use zeros for missing images.
        
        # Multi-task labels.
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'labels': {
                'admission_prediction': torch.tensor(int(row['admission']), dtype=torch.long),
                'length_of_stay': torch.tensor(int(row['length_of_stay']), dtype=torch.long),
                'severity_scoring': torch.tensor(int(row['severity_score']), dtype=torch.long)
            }
        }


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    task_metrics = {task: {'preds': [], 'labels': []} for task in ['admission_prediction', 'length_of_stay', 'severity_scoring']}
    
    for batch in tqdm(train_loader, desc='Training'):
        labels_dict = {task: batch['labels'][task].to(device) for task in batch['labels']}
        
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            pixel_values=batch['pixel_values'].to(device),
            labels=labels_dict
        )
        
        loss = outputs['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions.
        for task_name, task_logits in outputs['logits'].items():
            preds = torch.argmax(task_logits, dim=-1)
            task_metrics[task_name]['preds'].extend(preds.cpu().numpy())
            task_metrics[task_name]['labels'].extend(labels_dict[task_name].cpu().numpy())
    
    # Calculate accuracy for each task.
    task_accs = {}
    for task_name in task_metrics:
        acc = accuracy_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        task_accs[task_name] = acc
    
    return total_loss / len(train_loader), task_accs


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    task_metrics = {task: {'preds': [], 'labels': []} for task in ['admission_prediction', 'length_of_stay', 'severity_scoring']}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            labels_dict = {task: batch['labels'][task].to(device) for task in batch['labels']}
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                pixel_values=batch['pixel_values'].to(device),
                labels=labels_dict
            )
            
            total_loss += outputs['loss'].item()
            
            for task_name, task_logits in outputs['logits'].items():
                preds = torch.argmax(task_logits, dim=-1)
                task_metrics[task_name]['preds'].extend(preds.cpu().numpy())
                task_metrics[task_name]['labels'].extend(labels_dict[task_name].cpu().numpy())
    
    task_accs = {}
    for task_name in task_metrics:
        acc = accuracy_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        task_accs[task_name] = acc
    
    return total_loss / len(val_loader), task_accs


def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV-ED-Outcome Multi-task Training')
    parser.add_argument('--model_path', type=str, default='../../pytorch_model.bin')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--ecg_dir', type=str, required=True, help='ECG image directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mimic_outcome')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("MIMIC-IV-ED-Outcome training - multimodal multi-task")
    print("="*80 + "\n")
    
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path, dataset_name='MIMIC-IV-ED-Outcome', task_type='prognosis_prediction')
    model.freeze_encoders()
    model = model.to(args.device)
    
    tokenizer = SimpleEDTokenizer()
    train_dataset = MIMICOutcomeDataset(args.train_data, args.ecg_dir, tokenizer)
    val_dataset = MIMICOutcomeDataset(args.val_data, args.ecg_dir, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    optimizer = optim.AdamW(model.task_head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    best_avg_acc = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss, train_accs = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_accs = evaluate(model, val_loader, args.device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        for task, acc in train_accs.items():
            print(f"  {task}: {acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        for task, acc in val_accs.items():
            print(f"  {task}: {acc:.4f}")
        
        avg_acc = sum(val_accs.values()) / len(val_accs)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save({'model_state_dict': model.state_dict(), 'val_accs': val_accs},
                      os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  Saved best model (average accuracy: {avg_acc:.4f})")
    
    print(f"\nTraining complete. Best average accuracy: {best_avg_acc:.4f}")


if __name__ == '__main__':
    main()
