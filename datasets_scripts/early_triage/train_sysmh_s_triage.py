#!/usr/bin/env python3
"""
SYSMH-S-Triage 训练脚本
中山医院南院区急诊分诊 - 4分类任务
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
    """SYSMH-S-Triage 数据集"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 读取CSV数据
        self.data = pd.read_csv(data_path)
        print(f"加载 {len(self.data)} 条数据")
        
        # 标签映射: 1,2,3,4 -> 0,1,2,3
        self.data['label'] = self.data['label'] - 1
        
        # 类别分布
        label_counts = self.data['label'].value_counts().sort_index()
        print(f"类别分布: {dict(label_counts)}")
    
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
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
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
    """评估模型"""
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
                       help='预训练模型路径')
    parser.add_argument('--train_data', type=str, required=True,
                       help='训练数据路径 (CSV格式)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='验证数据路径 (CSV格式)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/sysmh_s_triage',
                       help='保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("SYSMH-S-Triage 训练 - 4分类急诊分诊")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"训练数据: {args.train_data}")
    print(f"验证数据: {args.val_data}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"设备: {args.device}")
    print("="*80 + "\n")
    
    # 加载模型
    print("[1/4] 加载模型...")
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path,
        dataset_name='SYSMH-S-Triage',
        task_type='early_triage'
    )
    model.freeze_encoders()  # 确保编码器冻结
    model = model.to(args.device)
    print("✓ 模型加载完成\n")
    
    # 加载数据
    print("[2/4] 加载数据...")
    tokenizer = SimpleEDTokenizer()
    train_dataset = SYSMHSTriageDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = SYSMHSTriageDataset(args.val_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("✓ 数据加载完成\n")
    
    # 设置优化器
    print("[3/4] 设置优化器...")
    optimizer = optim.AdamW(
        model.task_head.parameters(),  # 只优化任务头
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    print("✓ 优化器设置完成\n")
    
    # 训练
    print("[4/4] 开始训练...")
    print("-"*80)
    
    best_f1 = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-"*80)
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, args.device)
        
        # 验证
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, args.device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
        
        # 打印结果
        print(f"\n结果:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # 保存最佳模型
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
            print(f"  ✓ 保存最佳模型 (F1: {val_f1:.4f})")
            
            # 保存分类报告
            report = classification_report(
                val_labels,
                val_preds,
                target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                digits=4
            )
            with open(os.path.join(args.save_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
    
    # 保存训练历史
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 最终报告
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"最佳模型: Epoch {best_epoch}, F1: {best_f1:.4f}")
    print(f"模型保存在: {args.save_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

