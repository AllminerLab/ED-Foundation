#!/usr/bin/env python3
"""
MIMIC-IV多任务分类训练脚本
使用BEiT3模型的文本和图像编码器（冻结）+ 融合层（可训练）+ 多任务头（可训练）
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json
import logging
from datetime import datetime

# 导入自定义模块
from beit3_model import BEiT3MultiTaskModel, load_beit3_model_from_checkpoint
from mimic_dataset import create_mimic_dataloaders
from simple_tokenizer import SimpleTokenizer


class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=50,
        save_dir='checkpoints_multitask',
        use_amp=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.use_amp = use_amp
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 优化器：只优化可训练的参数（融合层和任务头）
        trainable_params = model.get_trainable_params()
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # 混合精度训练
        self.scaler = GradScaler(enabled=use_amp)
        
        self.logger.info(f"优化器: AdamW, lr={learning_rate}, wd={weight_decay}")
        self.logger.info(f"混合精度训练: {use_amp}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.save_dir, 'training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {name: 0.0 for name in self.model.task_names}
        task_preds = {name: [] for name in self.model.task_names}
        task_labels = {name: [] for name in self.model.task_names}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.max_epochs} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['images'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    labels=labels
                )
                
                loss = outputs['losses']['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            total_loss += loss.item()
            
            for task_name in self.model.task_names:
                task_loss = outputs['losses'][f'{task_name}_loss'].item()
                task_losses[task_name] += task_loss
                
                # 预测
                logits = outputs['logits'][task_name]
                preds = torch.argmax(logits, dim=1)
                task_preds[task_name].extend(preds.cpu().numpy())
                task_labels[task_name].extend(labels[task_name].cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 计算epoch统计
        avg_loss = total_loss / len(self.train_loader)
        
        metrics = {'loss': avg_loss}
        for task_name in self.model.task_names:
            avg_task_loss = task_losses[task_name] / len(self.train_loader)
            acc = accuracy_score(task_labels[task_name], task_preds[task_name])
            f1 = f1_score(task_labels[task_name], task_preds[task_name], average='binary', zero_division=0)
            
            metrics[f'{task_name}_loss'] = avg_task_loss
            metrics[f'{task_name}_acc'] = acc
            metrics[f'{task_name}_f1'] = f1
        
        return metrics
    
    @torch.no_grad()
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {name: 0.0 for name in self.model.task_names}
        task_preds = {name: [] for name in self.model.task_names}
        task_labels = {name: [] for name in self.model.task_names}
        task_probs = {name: [] for name in self.model.task_names}
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.max_epochs} [Val]')
        
        for batch in pbar:
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['images'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    labels=labels
                )
                
                loss = outputs['losses']['total_loss']
            
            # 统计
            total_loss += loss.item()
            
            for task_name in self.model.task_names:
                task_loss = outputs['losses'][f'{task_name}_loss'].item()
                task_losses[task_name] += task_loss
                
                # 预测
                logits = outputs['logits'][task_name]
                probs = torch.softmax(logits, dim=1)[:, 1]  # 正类概率
                preds = torch.argmax(logits, dim=1)
                
                task_preds[task_name].extend(preds.cpu().numpy())
                task_labels[task_name].extend(labels[task_name].cpu().numpy())
                task_probs[task_name].extend(probs.cpu().numpy())
        
        # 计算epoch统计
        avg_loss = total_loss / len(self.val_loader)
        
        metrics = {'loss': avg_loss}
        for task_name in self.model.task_names:
            avg_task_loss = task_losses[task_name] / len(self.val_loader)
            acc = accuracy_score(task_labels[task_name], task_preds[task_name])
            f1 = f1_score(task_labels[task_name], task_preds[task_name], average='binary', zero_division=0)
            
            # AUC（如果有正负样本）
            try:
                auc = roc_auc_score(task_labels[task_name], task_probs[task_name])
            except:
                auc = 0.0
            
            metrics[f'{task_name}_loss'] = avg_task_loss
            metrics[f'{task_name}_acc'] = acc
            metrics[f'{task_name}_f1'] = f1
            metrics[f'{task_name}_auc'] = auc
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'检查点已保存: {checkpoint_path}')
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f'✓ 最佳模型已保存: {best_path}')
    
    def train(self):
        """训练主循环"""
        self.logger.info("\n" + "="*80)
        self.logger.info("开始训练多任务模型")
        self.logger.info("="*80)
        
        best_val_f1 = 0.0
        
        for epoch in range(self.max_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 日志
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Epoch {epoch+1}/{self.max_epochs} 完成")
            self.logger.info(f"训练损失: {train_metrics['loss']:.4f}")
            self.logger.info(f"验证损失: {val_metrics['loss']:.4f}")
            
            # 每个任务的指标
            avg_val_f1 = 0.0
            for task_name in self.model.task_names:
                train_acc = train_metrics[f'{task_name}_acc']
                train_f1 = train_metrics[f'{task_name}_f1']
                val_acc = val_metrics[f'{task_name}_acc']
                val_f1 = val_metrics[f'{task_name}_f1']
                val_auc = val_metrics[f'{task_name}_auc']
                
                self.logger.info(
                    f"  {task_name}: "
                    f"Train Acc={train_acc:.4f}, F1={train_f1:.4f} | "
                    f"Val Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}"
                )
                avg_val_f1 += val_f1
            
            avg_val_f1 /= len(self.model.task_names)
            self.logger.info(f"平均验证F1: {avg_val_f1:.4f}")
            self.logger.info(f"{'='*80}\n")
            
            # 保存检查点
            is_best = avg_val_f1 > best_val_f1
            if is_best:
                best_val_f1 = avg_val_f1
            
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
        
        self.logger.info("\n训练完成！")
        self.logger.info(f"最佳平均F1: {best_val_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV多任务训练')
    
    # 数据参数
    parser.add_argument('--csv_path', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--ecg_root_dir', type=str, default='.', help='ECG图像根目录')
    parser.add_argument('--fold', type=int, default=0, help='验证集fold')
    
    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='BEiT3预训练checkpoint路径')
    parser.add_argument('--fusion_type', type=str, default='concat',
                       choices=['concat', 'attention', 'gated'],
                       help='融合类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度')
    
    # 其他
    parser.add_argument('--save_dir', type=str, default='checkpoints_multitask',
                       help='保存目录')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MIMIC-IV多任务分类训练")
    print("="*80)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建tokenizer
    print("\n[1/4] 创建tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=64010)
    print("✓ Tokenizer创建完成")
    
    # 创建数据加载器
    print("\n[2/4] 创建数据加载器...")
    train_loader, val_loader = create_mimic_dataloaders(
        csv_path=args.csv_path,
        ecg_root_dir=args.ecg_root_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=args.fold
    )
    print(f"✓ 训练集: {len(train_loader.dataset)} 样本")
    print(f"✓ 验证集: {len(val_loader.dataset)} 样本")
    
    # 创建模型
    print("\n[3/4] 创建模型...")
    
    # 创建timm视觉模型
    timm_model = timm.create_model('beit_base_patch16_224', pretrained=False, num_classes=0)
    
    # 创建BEiT3多任务模型并加载预训练权重
    model = load_beit3_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        timm_model=timm_model,
        device=device
    )
    
    # 创建训练器
    print("\n[4/4] 初始化训练器...")
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        save_dir=args.save_dir,
        use_amp=args.use_amp
    )
    print("✓ 训练器初始化完成")
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

