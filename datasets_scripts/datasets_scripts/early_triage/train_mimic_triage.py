#!/usr/bin/env python3
"""
MIMIC-IV-ED-Triage 训练脚本
MIMIC-IV急诊分诊 - 5分类任务
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class MIMICTriageDataset(Dataset):
    """MIMIC-IV-ED-Triage 数据集 (5分类)"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        
        # 标签映射: 1,2,3,4,5 -> 0,1,2,3,4
        self.data['label'] = self.data['label'] - 1
        
        print(f"加载 {len(self.data)} 条数据")
        label_counts = self.data['label'].value_counts().sort_index()
        print(f"类别分布: {dict(label_counts)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        
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
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs['logits'], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(train_loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs['loss'].item()
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted'), all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV-ED-Triage Training')
    parser.add_argument('--model_path', type=str, default='../../pytorch_model.bin')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mimic_triage')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("MIMIC-IV-ED-Triage 训练 - 5分类急诊分诊")
    print("="*80 + "\n")
    
    # 加载模型
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path,
        dataset_name='MIMIC-IV-ED-Triage',
        task_type='early_triage'
    )
    model.freeze_encoders()
    model = model.to(args.device)
    
    # 加载数据
    tokenizer = SimpleEDTokenizer()
    train_dataset = MIMICTriageDataset(args.train_data, tokenizer)
    val_dataset = MIMICTriageDataset(args.val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 优化器
    optimizer = optim.AdamW(model.task_head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 训练
    best_f1 = 0.0
    history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, args.device)
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1
            }, os.path.join(args.save_dir, 'best_model.pt'))
            
            report = classification_report(val_labels, val_preds,
                                          target_names=[f'Level {i+1}' for i in range(5)],
                                          digits=4)
            with open(os.path.join(args.save_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n训练完成！最佳F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()

