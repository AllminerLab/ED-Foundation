#!/usr/bin/env python3
"""
BEiT3-MoE with Timm - 掩码重建训练脚本

本脚本提供了完整的训练功能：
1. 模型创建和初始化
2. 预训练权重加载（从beit3_pre_para）
3. 混合精度训练 (AMP)
4. 梯度累积
5. 检查点管理
6. 学习率调度
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import json
from datetime import datetime
import time

# 禁用tokenizers的并行警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 导入模型
try:
    from beit3_timm import BEiT3WithMoEFromTimm, load_timm_model_weights
except ImportError:
    print("错误: 无法导入 beit3_timm 模块，请确保 beit3_timm.py 在同一目录")
    sys.exit(1)

# 导入timm
try:
    import timm
except ImportError:
    print("错误: 无法导入 timm，请运行: pip install timm")
    sys.exit(1)


class BEiT3TimmTrainer:
    """BEiT3-Timm 训练器，支持掩码重建任务"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        max_epochs: int = 10,
        save_dir: str = 'checkpoints_beit3_timm',
        log_interval: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 0.5,
        use_amp: bool = True,
        use_compile: bool = False,
        compile_mode: str = 'reduce-overhead',
        channels_last: bool = True
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.channels_last = channels_last
        
        # 设置模型
        self.model = model.to(device)
        
        # 应用channels_last格式
        if self.channels_last:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                print("[Trainer] ✓ Channels Last 已启用")
            except Exception as e:
                print(f"[Trainer] ✗ Channels Last 启用失败: {e}")
        
        # torch.compile 优化
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                compile_mode = getattr(self, 'compile_mode', 'reduce-overhead')
                print(f"[Trainer] 正在编译模型（模式: {compile_mode}，首次运行较慢）...")
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    fullgraph=False
                )
                print("[Trainer] ✓ torch.compile 已启用")
            except Exception as e:
                print(f"[Trainer] ✗ torch.compile 失败: {e}")
                self.use_compile = False
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 设置日志
        self._setup_logging()
        
        # 设置优化器和调度器
        self._setup_optimizer_and_scheduler(learning_rate, weight_decay)
        
        # 混合精度
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("[Trainer] ✓ 混合精度训练 (AMP) 已启用")
        
        # 性能监控
        self.batch_times = []
        self.data_times = []
    
    def _setup_logging(self):
        """设置日志"""
        os.makedirs(self.save_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_optimizer_and_scheduler(self, learning_rate: float, weight_decay: float):
        """设置优化器和学习率调度器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.logger.info(f"[Optimizer] AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=learning_rate * 0.01
        )
        self.logger.info(f"[Scheduler] CosineAnnealingLR (T_max={self.max_epochs})")
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """准备批次数据（优化版：减少内存拷贝）"""
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # 使用non_blocking异步传输加速
                tensor = value.to(self.device, non_blocking=True)
                # 只对图像应用channels_last格式
                if self.channels_last and key == 'image_pixel_values' and tensor.dim() == 4:
                    tensor = tensor.to(memory_format=torch.channels_last)
                prepared_batch[key] = tensor
            else:
                prepared_batch[key] = value
        return prepared_batch
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch（优化版：减少冗余检查）"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        num_batches = 0
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f'Epoch {epoch+1}/{self.max_epochs} [Train]',
            leave=False
        )
        
        accumulated_loss = 0.0
        accumulated_samples = 0
        accumulation_step_count = 0  # 跟踪实际的梯度累积次数
        batch_start_time = time.time()
        data_load_time = 0
        
        # 性能优化：减少频繁的 .item() 调用
        log_losses = []  # 累积损失用于日志
        
        for batch_idx, batch in enumerate(pbar):
            data_load_time = time.time() - batch_start_time
            
            # 准备数据（使用non_blocking加速）
            batch = self._prepare_batch(batch)
            
            # 快速检查：只在前几个batch进行详细验证
            if batch_idx < 3 or batch_idx % 100 == 0:
                # 详细的输入数据检查
                if torch.isnan(batch['image_pixel_values']).any() or torch.isinf(batch['image_pixel_values']).any():
                    self.logger.warning(f"Batch {batch_idx}: 图像数据包含NaN或Inf，跳过此批次")
                    # 清理梯度累积状态
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_loss = 0.0
                    accumulated_samples = 0
                    accumulation_step_count = 0
                    batch_start_time = time.time()
                    continue
                
                # 检查图像标签
                if batch['image_labels'] is not None:
                    img_label_shape = batch['image_labels'].shape
                    if len(img_label_shape) != 5 or img_label_shape[1] != 196:
                        self.logger.error(f"Batch {batch_idx}: 图像标签形状错误 {img_label_shape}")
                        # 清理梯度累积状态
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulated_loss = 0.0
                        accumulated_samples = 0
                        accumulation_step_count = 0
                        batch_start_time = time.time()
                        continue
                    
                    if torch.isnan(batch['image_labels']).any() or torch.isinf(batch['image_labels']).any():
                        self.logger.warning(f"Batch {batch_idx}: 图像标签包含NaN或Inf")
                        # 清理梯度累积状态
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulated_loss = 0.0
                        accumulated_samples = 0
                        accumulation_step_count = 0
                        batch_start_time = time.time()
                        continue
            
            # 前向传播（混合精度）
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    image=batch['image_pixel_values'],
                    input_ids=batch['text_input_ids'],
                    attention_mask=batch['text_attention_mask'],
                    labels=batch['labels'],
                    image_labels=batch['image_labels'],
                    data_type=batch.get('data_type')  # 传递数据类型信息
                )
                
                loss = outputs['losses']['total_loss'] / self.gradient_accumulation_steps
                
                # 提取分项损失（用于显示）
                text_loss_val = outputs['losses'].get('text_loss', torch.tensor(0.0))
                image_loss_val = outputs['losses'].get('image_loss', torch.tensor(0.0))
                moe_loss_val = outputs['losses'].get('moe_aux_loss', torch.tensor(0.0))
            
            # 快速NaN检查（避免不必要的详细日志）
            if torch.isnan(loss) or torch.isinf(loss):
                # 只在检测到NaN时才进行详细检查和日志
                losses_dict = outputs['losses']
                self.logger.error(f"Batch {batch_idx}: 总损失为NaN或Inf")
                for loss_name, loss_value in losses_dict.items():
                    if isinstance(loss_value, torch.Tensor) and (torch.isnan(loss_value).any() or torch.isinf(loss_value).any()):
                        self.logger.error(f"  {loss_name} 为 NaN/Inf")
                # 清理梯度累积状态
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                accumulated_samples = 0
                accumulation_step_count = 0
                batch_start_time = time.time()
                continue
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 累积（避免频繁的.item()调用）
            accumulated_loss += loss.detach() * self.gradient_accumulation_steps
            accumulated_samples += batch['image_pixel_values'].size(0)
            accumulation_step_count += 1
            
            # 梯度累积：每gradient_accumulation_steps步更新一次
            if accumulation_step_count >= self.gradient_accumulation_steps:
                # Unscale梯度以便检查和裁剪
                self.scaler.unscale_(self.optimizer)
                
                # ========== 改进的梯度范数计算 ==========
                # 手动计算梯度范数，更鲁棒地处理NaN
                skip_update = False
                grad_norm = 0.0
                has_grad_nan = False
                total_norm = 0.0
                
                for p in self.model.parameters():
                    if p.grad is not None:
                        # 检查梯度是否包含NaN/Inf
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            has_grad_nan = True
                            self.logger.error(f"Batch {batch_idx}: 参数梯度包含NaN/Inf！")
                            skip_update = True
                            break
                        
                        # 累积梯度范数（L2范数）
                        param_norm = torch.norm(p.grad.data)
                        total_norm += param_norm ** 2
                
                if not has_grad_nan:
                    # 计算总梯度范数
                    grad_norm = torch.sqrt(total_norm).item() if total_norm > 0 else 0.0
                    
                    # 检查计算出的梯度范数是否为NaN或Inf
                    if np.isnan(grad_norm) or np.isinf(grad_norm):
                        self.logger.error(f"Batch {batch_idx}: 计算的梯度范数为NaN/Inf ({grad_norm})")
                        skip_update = True
                    elif grad_norm > 10.0:
                        self.logger.warning(f"Batch {batch_idx}: 梯度范数较大 ({grad_norm:.2f})")
                    
                    # 手动梯度裁剪（如果梯度范数有效且过大）
                    if grad_norm > 0 and not skip_update and grad_norm > self.max_grad_norm:
                        scale_factor = self.max_grad_norm / max(grad_norm, 1e-8)
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(scale_factor)
                        grad_norm = self.max_grad_norm
                
                # 如果梯度有效，更新参数
                if not skip_update:
                    # 记录更新前的一些参数统计（用于调试）
                    if batch_idx < 10:  # 简化：只在前10个批次记录
                        with torch.no_grad():
                            param_norms = []
                            for name, param in self.model.named_parameters():
                                if param.requires_grad and param.grad is not None:
                                    param_norms.append(param.data.norm().item())
                            if param_norms:
                                avg_param_norm = sum(param_norms) / len(param_norms)
                                max_param_norm = max(param_norms)
                                self.logger.debug(
                                    f"Batch {batch_idx} [更新前]: "
                                    f"grad_norm={grad_norm:.4f}, "
                                    f"avg_param_norm={avg_param_norm:.4f}, "
                                    f"max_param_norm={max_param_norm:.4f}"
                                )
                    
                    self.scaler.step(self.optimizer)
                    
                    # 更新后检查参数是否出现NaN（关键检查！）
                    if batch_idx < 10:
                        with torch.no_grad():
                            params_have_nan = False
                            for name, param in self.model.named_parameters():
                                if torch.isnan(param).any() or torch.isinf(param).any():
                                    self.logger.error(
                                        f"Batch {batch_idx} [更新后]: 参数 {name} 出现 NaN/Inf！"
                                    )
                                    params_have_nan = True
                            
                            if params_have_nan:
                                self.logger.error(
                                    f"⚠️  参数更新后出现NaN，下一个batch可能会失败！"
                                    f"建议：降低学习率或增加梯度裁剪力度"
                                )
                    
                    # 更新统计（只在成功时）
                    loss_val = accumulated_loss.item()
                    total_loss += loss_val
                    total_samples += accumulated_samples
                    num_batches += 1
                    log_losses.append(loss_val)
                else:
                    # ⚠️ 关键：即使skip_update=True，也必须调用scaler.step()
                    # GradScaler的使用规则：unscale_后必须调用step()
                    # 否则会抛出 AssertionError: No inf checks were recorded
                    self.scaler.step(self.optimizer)
                
                # 无论是否更新参数，都要更新scaler和清空梯度
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                # 重置累积变量
                accumulated_loss = 0.0
                accumulated_samples = 0
                accumulation_step_count = 0
                
                # 更新进度条（每次更新，显示实时loss）
                batch_time = time.time() - batch_start_time
                self.batch_times.append(batch_time)
                self.data_times.append(data_load_time)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                current_loss = loss_val  # 当前batch的loss（实时）
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0  # 累积平均loss
                avg_batch_time = np.mean(self.batch_times[-100:]) if self.batch_times else 0
                
                # 实时更新进度条，显示当前batch的loss、分项loss和平均loss
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',  # 当前batch total loss
                    'Text': f'{text_loss_val.item() if isinstance(text_loss_val, torch.Tensor) else text_loss_val:.4f}',  # 文本loss
                    'Image': f'{image_loss_val.item() if isinstance(image_loss_val, torch.Tensor) else image_loss_val:.4f}',  # 图像loss
                    'AvgLoss': f'{avg_loss:.4f}',   # 平均total loss
                    'LR': f'{current_lr:.2e}',
                    'Batch': f'{avg_batch_time:.2f}s'
                })
                
                # 详细日志（减少频率）
                if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    current_lr = self.optimizer.param_groups[0]['lr']
                    # 只记录必要信息
                    self.logger.info(
                        f'Epoch {epoch+1}, Batch {batch_idx}, '
                        f'Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, '
                        f'Samples: {total_samples}'
                    )
            
            batch_start_time = time.time()
        
        # 处理最后一个不完整的梯度累积
        if accumulated_samples > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            # 无论梯度是否有效，都必须调用step()
            self.scaler.step(self.optimizer)
            
            # 只有在梯度有效时才更新统计
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                total_loss += accumulated_loss.item()
                total_samples += accumulated_samples
                num_batches += 1
            
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        # 更新学习率
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 性能统计
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            throughput = total_samples / sum(self.batch_times) if sum(self.batch_times) > 0 else 0
            self.logger.info(
                f'\n[Epoch {epoch+1}] 平均批次时间: {avg_batch_time:.3f}s, '
                f'总样本数: {total_samples}, 吞吐量: {throughput:.1f} samples/s'
            )
        
        self.batch_times = []
        self.data_times = []
        
        return {
            'train_loss': avg_loss,
            'train_samples': total_samples,
            'train_batches': num_batches
        }
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        num_batches = 0
        
        pbar = tqdm(
            self.val_dataloader,
            desc=f'Epoch {epoch+1}/{self.max_epochs} [Val]',
            leave=False
        )
        
        for batch in pbar:
            # 准备数据
            batch = self._prepare_batch(batch)
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    image=batch['image_pixel_values'],
                    input_ids=batch['text_input_ids'],
                    attention_mask=batch['text_attention_mask'],
                    labels=batch['labels'],
                    image_labels=batch['image_labels'],
                    data_type=batch.get('data_type')  # 传递数据类型信息
                )
                
                loss = outputs['losses']['total_loss']
            
            total_loss += loss.item()
            total_samples += batch['image_pixel_values'].size(0)
            num_batches += 1
            
            if num_batches % 10 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                pbar.set_postfix({'Val Loss': f'{avg_loss:.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        self.logger.info(
            f'Validation Epoch {epoch+1}: Loss: {avg_loss:.4f}, Samples: {total_samples}'
        )
        
        return {
            'val_loss': avg_loss,
            'val_samples': total_samples,
            'val_batches': num_batches
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        # 获取模型状态的更安全方法
        try:
            # 首先尝试直接获取state_dict
            model_state = self.model.state_dict()
        except Exception as e:
            self.logger.warning(f"获取模型state_dict失败，尝试备用方法: {e}")
            try:
                # 如果模型被包装（如DDP、torch.compile等），尝试访问module
                if hasattr(self.model, 'module'):
                    model_state = self.model.module.state_dict()
                else:
                    model_state = self.model.state_dict()
            except Exception as e2:
                self.logger.error(f"无法获取模型state_dict: {e2}")
                model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'metrics': metrics,
            'config': {
                'max_epochs': self.max_epochs,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'use_amp': self.use_amp,
                'use_compile': self.use_compile,
                'channels_last': self.channels_last
            }
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'检查点已保存: {checkpoint_path}')
        
        if is_best:
            best_checkpoint_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f'✓ 新的最佳模型已保存: {best_checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.logger.info("模型状态加载成功")
        except Exception as e:
            self.logger.warning(f"模型状态加载失败: {e}")
            return 0
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("优化器状态加载成功")
        except Exception as e:
            self.logger.warning(f"优化器状态加载失败: {e}")
        
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("调度器状态加载成功")
        except Exception as e:
            self.logger.warning(f"调度器状态加载失败: {e}")
        
        if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.logger.info("GradScaler状态加载成功")
            except Exception as e:
                self.logger.warning(f"GradScaler状态加载失败: {e}")
        
        self.logger.info(f'检查点已加载: {checkpoint_path}')
        return checkpoint['epoch'] + 1
    
    def train(self, resume_from: Optional[str] = None):
        """训练主循环"""
        self.logger.info("\n" + "="*80)
        self.logger.info("开始训练")
        self.logger.info("="*80)
        self.logger.info(f"训练配置:")
        self.logger.info(f"  Epochs: {self.max_epochs}")
        self.logger.info(f"  梯度累积步数: {self.gradient_accumulation_steps}")
        self.logger.info(f"  混合精度 (AMP): {self.use_amp}")
        self.logger.info(f"  torch.compile: {self.use_compile}")
        self.logger.info(f"  Channels Last: {self.channels_last}")
        self.logger.info("="*80 + "\n")
        
        best_val_loss = float('inf')
        start_epoch = 0
        
        # 恢复检查点
        if resume_from and os.path.exists(resume_from):
            self.logger.info(f"从检查点恢复: {resume_from}")
            start_epoch = self.load_checkpoint(resume_from)
            best_val_loss = float('inf')
        
        train_start_time = time.time()
        
        for epoch in range(start_epoch, self.max_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 合并指标
            metrics = {**train_metrics, **val_metrics}
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Epoch {epoch+1}/{self.max_epochs} 完成")
            self.logger.info(f"  训练损失: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"  验证损失: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"  Epoch时间: {epoch_time:.2f}s")
            self.logger.info(f"{'='*80}\n")
            
            # 保存检查点
            is_best = val_metrics['val_loss'] < best_val_loss
            self.save_checkpoint(epoch, metrics, is_best=is_best)
            
            if is_best:
                best_val_loss = val_metrics['val_loss']
        
        total_time = time.time() - train_start_time
        self.logger.info("\n" + "="*80)
        self.logger.info("训练完成！")
        self.logger.info(f"总耗时: {total_time/3600:.2f} 小时")
        self.logger.info("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BEiT3-MoE with Timm 掩码重建训练脚本'
    )
    
    # 数据集参数
    parser.add_argument('--batch_size', type=int, default=12, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数（增加以加速）')
    parser.add_argument('--max_samples', type=int, default=1000000, help='最大样本数')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='每个worker预取批次数')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='使用固定内存')
    
    # 数据分割比例参数（新增）
    parser.add_argument('--val_split', type=float, default=0.2, 
                       help='验证集占比（0-1之间），默认0.2表示验证集为总数的20%%')
    
    # 多模态数据比例参数（新增）
    parser.add_argument('--multimodal_ratio', type=float, default=0.6, 
                       help='多模态（图文）数据比例，默认0.6')
    parser.add_argument('--text_only_ratio', type=float, default=0.2, 
                       help='纯文本数据比例，默认0.2')
    parser.add_argument('--image_only_ratio', type=float, default=0.2, 
                       help='纯图像数据比例，默认0.2')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                       help='学习率（进一步降低以提高稳定性）')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='梯度累积步数（增加以节省显存并提高稳定性）')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, 
                       help='梯度裁剪的最大范数（降低以提高稳定性）')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true', default=True, 
                       help='使用混合精度')
    parser.add_argument('--use_compile', action='store_true', default=False, 
                       help='使用torch.compile加速（PyTorch 2.0+）')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile优化模式')
    parser.add_argument('--channels_last', action='store_true', default=True, 
                       help='使用channels_last格式（加速卷积）')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=True,
                       help='启用cuDNN benchmark以自动寻找最优算法')
    
    # 检查点参数
    parser.add_argument('--save_dir', type=str, default='checkpoints_beit3_timm', 
                       help='保存目录')
    parser.add_argument('--log_interval', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--resume_from', type=str, default=None, help='从检查点恢复')
    parser.add_argument('--load_pretrained', type=str, 
                       default='beit3_pre_para/pytorch_model.bin',
                       help='加载预训练权重')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BEiT3-MoE with Timm - 掩码重建训练（优化版）")
    print("="*80)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 性能优化设置
    if args.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[优化] ✓ cuDNN benchmark 已启用")
    
    # 设置数据加载优化
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("[优化] ✓ 多进程共享策略已设置")
    
    # ==================== 创建模型 ====================
    print("\n[1/4] 创建模型...")
    
    base_model = timm.create_model('beit_base_patch16_224', pretrained=False, num_classes=0)
    
    if hasattr(base_model, 'embed_dim'):
        embed_dim = base_model.embed_dim
    elif hasattr(base_model, 'num_features'):
        embed_dim = base_model.num_features
    else:
        embed_dim = 768
    
    num_layers = len(base_model.blocks) if hasattr(base_model, 'blocks') else 12
    num_heads = base_model.blocks[0].attn.num_heads if (hasattr(base_model, 'blocks') and 
                                                         len(base_model.blocks) > 0) else 12
    
    print(f"  模型配置: embed_dim={embed_dim}, num_layers={num_layers}, num_heads={num_heads}")
    
    model = BEiT3WithMoEFromTimm(
        timm_model=base_model,
        vocab_size=64010,
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embed_dim
    )
    
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== 加载预训练权重 ====================
    print("\n[2/4] 加载预训练权重...")
    
    if args.load_pretrained and os.path.exists(args.load_pretrained):
        print(f"  从 {args.load_pretrained} 加载权重...")
        load_timm_model_weights(model, args.load_pretrained)
    else:
        print(f"  警告: 预训练权重不存在 ({args.load_pretrained})")
    
    # ==================== 创建数据集 ====================
    print("\n[3/4] 创建数据集...")
    
    try:
        from mixed_modal_dataset_optimized import (
            create_mixed_modal_train_val_dataloaders_optimized,
            MixedModalConfigOptimized
        )
        
        # 创建训练配置
        train_config = MixedModalConfigOptimized(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            shuffle=True,
            multimodal_ratio=args.multimodal_ratio,
            text_only_ratio=args.text_only_ratio,
            image_only_ratio=args.image_only_ratio
        )
        
        # 创建验证配置（根据 val_split 计算验证集大小）
        val_config = None
        if args.val_split > 0 and args.max_samples:
            # 计算验证集和训练集的样本数
            val_samples = int(args.max_samples * args.val_split)
            train_samples = args.max_samples - val_samples
            
            # 创建验证数据配置
            val_config = MixedModalConfigOptimized(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=val_samples,
                shuffle=True,
                multimodal_ratio=args.multimodal_ratio,
                text_only_ratio=args.text_only_ratio,
                image_only_ratio=args.image_only_ratio
            )
            
            # 更新训练配置为训练样本数
            train_config = MixedModalConfigOptimized(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=train_samples,
                shuffle=True,
                multimodal_ratio=args.multimodal_ratio,
                text_only_ratio=args.text_only_ratio,
                image_only_ratio=args.image_only_ratio
            )
        
        # 创建数据加载器
        train_dataloader, val_dataloader = create_mixed_modal_train_val_dataloaders_optimized(
            train_config=train_config,
            val_config=val_config
        )
        print(f"  ✓ 数据加载器已创建")
        if args.val_split > 0 and args.max_samples:
            print(f"    - 总样本: {args.max_samples}")
            print(f"    - 训练集: {train_samples} ({(1-args.val_split)*100:.1f}%)")
            print(f"    - 验证集: {val_samples} ({args.val_split*100:.1f}%)")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        print("  请确保 mixed_modal_dataset_optimized.py 可用")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ==================== 创建训练器 ====================
    print("\n[4/4] 初始化训练器...")
    
    trainer = BEiT3TimmTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        channels_last=args.channels_last
    )
    
    print("  ✓ 训练器初始化完成")
    
    # ==================== 开始训练 ====================
    trainer.train(resume_from=args.resume_from)


if __name__ == '__main__':
    main()
