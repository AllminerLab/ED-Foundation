#!/usr/bin/env python3
"""
SYSMH-ED-MD 训练脚本
中山医院急诊医疗决策 - 纯文本多任务
任务: 1)影像检查 2)实验室检查 3)留观 4)专科会诊
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class SYSMHMDDataset(Dataset):
    """SYSMH-ED-MD 数据集 - 医疗决策四任务"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        print(f"加载 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        encoding = self.tokenizer(str(row['text']), max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': {
                'imaging_exam': torch.tensor(int(row['imaging_exam']), dtype=torch.long),
                'lab_tests': torch.tensor(int(row['lab_tests']), dtype=torch.long),
                'observation': torch.tensor(int(row['observation']), dtype=torch.long),
                'specialist_consultation': torch.tensor(int(row['specialist_consultation']), dtype=torch.long)
            }
        }


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    task_metrics = {task: {'preds': [], 'labels': [], 'probs': []} 
                   for task in ['imaging_exam', 'lab_tests', 'observation', 'specialist_consultation']}
    
    for batch in tqdm(train_loader, desc='Training'):
        labels_dict = {task: batch['labels'][task].to(device) for task in batch['labels']}
        
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=labels_dict
        )
        
        loss = outputs['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        for task_name, task_logits in outputs['logits'].items():
            probs = torch.softmax(task_logits, dim=-1)[:, 1]
            preds = torch.argmax(task_logits, dim=-1)
            task_metrics[task_name]['preds'].extend(preds.cpu().numpy())
            task_metrics[task_name]['labels'].extend(labels_dict[task_name].cpu().numpy())
            task_metrics[task_name]['probs'].extend(probs.cpu().numpy())
    
    task_results = {}
    for task_name in task_metrics:
        acc = accuracy_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        f1 = f1_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        auc = roc_auc_score(task_metrics[task_name]['labels'], task_metrics[task_name]['probs'])
        task_results[task_name] = {'acc': acc, 'f1': f1, 'auc': auc}
    
    return total_loss / len(train_loader), task_results


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    task_metrics = {task: {'preds': [], 'labels': [], 'probs': []} 
                   for task in ['imaging_exam', 'lab_tests', 'observation', 'specialist_consultation']}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            labels_dict = {task: batch['labels'][task].to(device) for task in batch['labels']}
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=labels_dict
            )
            
            total_loss += outputs['loss'].item()
            
            for task_name, task_logits in outputs['logits'].items():
                probs = torch.softmax(task_logits, dim=-1)[:, 1]
                preds = torch.argmax(task_logits, dim=-1)
                task_metrics[task_name]['preds'].extend(preds.cpu().numpy())
                task_metrics[task_name]['labels'].extend(labels_dict[task_name].cpu().numpy())
                task_metrics[task_name]['probs'].extend(probs.cpu().numpy())
    
    task_results = {}
    for task_name in task_metrics:
        acc = accuracy_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        f1 = f1_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        auc = roc_auc_score(task_metrics[task_name]['labels'], task_metrics[task_name]['probs'])
        task_results[task_name] = {'acc': acc, 'f1': f1, 'auc': auc}
    
    return total_loss / len(val_loader), task_results


def main():
    parser = argparse.ArgumentParser(description='SYSMH-ED-MD Training')
    parser.add_argument('--model_path', type=str, default='../../pytorch_model.bin')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/sysmh_md')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("SYSMH-ED-MD 训练 - 医疗决策四任务")
    print("="*80 + "\n")
    
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path, dataset_name='SYSMH-ED-MD', task_type='full_process_decision')
    model.freeze_encoders()
    model = model.to(args.device)
    
    tokenizer = SimpleEDTokenizer()
    train_dataset = SYSMHMDDataset(args.train_data, tokenizer)
    val_dataset = SYSMHMDDataset(args.val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    optimizer = optim.AdamW(model.task_head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    best_avg_f1 = 0.0
    history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss, train_results = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_results = evaluate(model, val_loader, args.device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        for task, metrics in train_results.items():
            print(f"  {task}: Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        for task, metrics in val_results.items():
            print(f"  {task}: Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        
        avg_f1 = sum([m['f1'] for m in val_results.values()]) / len(val_results)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in train_results.items()},
            'val_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in val_results.items()},
            'avg_f1': float(avg_f1)
        })
        
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            torch.save({'model_state_dict': model.state_dict(), 'val_results': val_results},
                      os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ✓ 保存最佳模型 (平均F1: {avg_f1:.4f})")
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n训练完成！最佳平均F1: {best_avg_f1:.4f}")


if __name__ == '__main__':
    main()

