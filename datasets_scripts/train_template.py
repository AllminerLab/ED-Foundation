#!/usr/bin/env python3
"""
BEiT3-ED Foundation Model - 通用训练脚本模板
线性探测训练（编码器冻结）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import json
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class EDDataset(Dataset):
    """急诊数据集基类 - 需要根据具体数据集实现"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, data_path):
        """加载数据 - 子类实现"""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本 - 子类实现"""
        raise NotImplementedError


def train_linear_probing(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=1e-3,
    device='cuda',
    save_dir='./checkpoints'
):
    """
    线性探测训练
    
    Args:
        model: BEiT3EDFoundationModel
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        save_dir: 保存目录
    """
    
    # 确保编码器冻结
    model.freeze_encoders()
    model = model.to(device)
    
    # 只优化任务头参数
    optimizer = optim.AdamW(
        model.task_head.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("开始线性探测训练")
    print("="*80)
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"批次大小: {train_loader.batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {num_epochs}")
    print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            # 移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            
            if isinstance(logits, dict):
                # 多任务：计算第一个任务的准确率
                task_name = list(logits.keys())[0]
                preds = torch.argmax(logits[task_name], dim=-1)
                train_correct += (preds == labels[task_name]).sum().item()
            else:
                # 单任务
                preds = torch.argmax(logits, dim=-1)
                train_correct += (preds == labels).sum().item()
            
            train_total += labels.size(0) if not isinstance(labels, dict) else list(labels.values())[0].size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                pixel_values = batch.get('pixel_values')
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                val_loss += loss.item()
                
                if isinstance(logits, dict):
                    task_name = list(logits.keys())[0]
                    preds = torch.argmax(logits[task_name], dim=-1)
                    val_correct += (preds == labels[task_name]).sum().item()
                else:
                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == labels).sum().item()
                
                val_total += labels.size(0) if not isinstance(labels, dict) else list(labels.values())[0].size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        # 打印结果
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            save_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, save_path)
            print(f'  ✓ 保存最佳模型: {save_path} (Val Acc: {val_acc:.2f}%)')
    
    print("\n" + "="*80)
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print("="*80 + "\n")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='BEiT3-ED Linear Probing Training')
    
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--task_type', type=str, required=True,
                       choices=['early_triage', 'prognosis_prediction', 'full_process_decision'],
                       help='任务类型')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val_data', type=str, required=True, help='验证数据路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 加载模型
    print("加载预训练模型...")
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path,
        dataset_name=args.dataset_name,
        task_type=args.task_type
    )
    
    # 创建数据集（需要根据具体数据集实现）
    tokenizer = SimpleEDTokenizer()
    
    # 这里需要根据具体数据集实现数据加载
    print("加载数据...")
    # train_dataset = YourDataset(args.train_data, tokenizer)
    # val_dataset = YourDataset(args.val_data, tokenizer)
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 训练
    # train_linear_probing(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=args.num_epochs,
    #     learning_rate=args.learning_rate,
    #     device=args.device,
    #     save_dir=args.save_dir
    # )
    
    print("请根据具体数据集实现数据加载部分")


if __name__ == '__main__':
    main()

