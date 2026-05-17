#!/usr/bin/env python3
"""
MIMIC-IV-EXT-MDS-ED 训练脚本
多模态(文本+CXR+ECG)多任务学习
任务: 1)机械通气 2)ICU转诊 3)7天死亡率 4)28天死亡率
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class MIMICMDSDataset(Dataset):
    """MIMIC-IV-EXT-MDS-ED 多模态多任务数据集"""
    def __init__(self, data_path, image_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_dir = image_dir
        self.data = pd.read_csv(data_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"加载 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 文本
        encoding = self.tokenizer(str(row['text']), max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
        
        # 图像 (CXR 或 ECG)
        image_path = os.path.join(self.image_dir, row['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.transform(image)
        except:
            pixel_values = torch.zeros(3, 224, 224)
        
        # 四个任务标签
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'labels': {
                'mechanical_ventilation': torch.tensor(int(row['mechanical_ventilation']), dtype=torch.long),
                'icu_stay': torch.tensor(int(row['icu_stay']), dtype=torch.long),
                'mortality_7d': torch.tensor(int(row['mortality_7d']), dtype=torch.long),
                'mortality_28d': torch.tensor(int(row['mortality_28d']), dtype=torch.long)
            }
        }


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    task_metrics = {task: {'preds': [], 'labels': [], 'probs': []} 
                   for task in ['mechanical_ventilation', 'icu_stay', 'mortality_7d', 'mortality_28d']}
    
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
        
        for task_name, task_logits in outputs['logits'].items():
            probs = torch.softmax(task_logits, dim=-1)[:, 1]
            preds = torch.argmax(task_logits, dim=-1)
            task_metrics[task_name]['preds'].extend(preds.cpu().numpy())
            task_metrics[task_name]['labels'].extend(labels_dict[task_name].cpu().numpy())
            task_metrics[task_name]['probs'].extend(probs.cpu().numpy())
    
    task_results = {}
    for task_name in task_metrics:
        acc = accuracy_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        auc = roc_auc_score(task_metrics[task_name]['labels'], task_metrics[task_name]['probs'])
        task_results[task_name] = {'acc': acc, 'auc': auc}
    
    return total_loss / len(train_loader), task_results


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    task_metrics = {task: {'preds': [], 'labels': [], 'probs': []} 
                   for task in ['mechanical_ventilation', 'icu_stay', 'mortality_7d', 'mortality_28d']}
    
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
                probs = torch.softmax(task_logits, dim=-1)[:, 1]
                preds = torch.argmax(task_logits, dim=-1)
                task_metrics[task_name]['preds'].extend(preds.cpu().numpy())
                task_metrics[task_name]['labels'].extend(labels_dict[task_name].cpu().numpy())
                task_metrics[task_name]['probs'].extend(probs.cpu().numpy())
    
    task_results = {}
    for task_name in task_metrics:
        acc = accuracy_score(task_metrics[task_name]['labels'], task_metrics[task_name]['preds'])
        auc = roc_auc_score(task_metrics[task_name]['labels'], task_metrics[task_name]['probs'])
        task_results[task_name] = {'acc': acc, 'auc': auc}
    
    return total_loss / len(val_loader), task_results


def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV-EXT-MDS-ED Training')
    parser.add_argument('--model_path', type=str, default='../../pytorch_model.bin')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True, help='CXR/ECG图像目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mimic_mds')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("MIMIC-IV-EXT-MDS-ED 训练 - 多模态多任务急诊决策")
    print("="*80 + "\n")
    
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path, dataset_name='MIMIC-IV-EXT-MDS-ED', task_type='full_process_decision')
    model.freeze_encoders()
    model = model.to(args.device)
    
    tokenizer = SimpleEDTokenizer()
    train_dataset = MIMICMDSDataset(args.train_data, args.image_dir, tokenizer)
    val_dataset = MIMICMDSDataset(args.val_data, args.image_dir, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    optimizer = optim.AdamW(model.task_head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    best_avg_auc = 0.0
    history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss, train_results = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_results = evaluate(model, val_loader, args.device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        for task, metrics in train_results.items():
            print(f"  {task}: Acc={metrics['acc']:.4f}, AUC={metrics['auc']:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        for task, metrics in val_results.items():
            print(f"  {task}: Acc={metrics['acc']:.4f}, AUC={metrics['auc']:.4f}")
        
        avg_auc = sum([m['auc'] for m in val_results.values()]) / len(val_results)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in train_results.items()},
            'val_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in val_results.items()},
            'avg_auc': float(avg_auc)
        })
        
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            torch.save({'model_state_dict': model.state_dict(), 'val_results': val_results},
                      os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ✓ 保存最佳模型 (平均AUC: {avg_auc:.4f})")
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n训练完成！最佳平均AUC: {best_avg_auc:.4f}")


if __name__ == '__main__':
    main()


