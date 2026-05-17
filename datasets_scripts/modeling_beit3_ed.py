#!/usr/bin/env python3
"""
BEiT3-ED Foundation Model - Emergency Department Foundation Model
急诊科通用基础模型
General-purpose foundation model for the Emergency Department.

支持三大类任务：
Supports three major task categories:
1. 早期急诊分诊 (Early ED Triage)
2. 预后预测 (Prognosis Prediction)
3. 全流程急诊决策 (Full-Process ED Decision)

训练策略: 线性探测 (Linear Probing) - 编码器冻结，只训练任务头
Training strategy: Linear Probing - freeze encoders and train only task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
import timm
import json


class TextEncoder(nn.Module):
    """文本编码器 - 基于Transformer / Text encoder based on Transformer."""
    def __init__(self, vocab_size=64010, embed_dim=768, num_layers=12, num_heads=12, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embedding_layer_norm = nn.LayerNorm(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        embeddings = self.embedding_layer_norm(token_embeds + position_embeds)
        
        if attention_mask is not None:
            mask = (attention_mask == 0)
        else:
            mask = None
        
        hidden_states = self.transformer_encoder(embeddings, src_key_padding_mask=mask)
        text_features = hidden_states[:, 0, :]
        
        return text_features


class VisionEncoder(nn.Module):
    """视觉编码器 - 基于BEiT / Vision encoder based on BEiT."""
    def __init__(self, timm_model):
        super().__init__()
        self.backbone = timm_model
        
    def forward(self, images):
        vision_features = self.backbone(images)
        return vision_features


class MultimodalFusion(nn.Module):
    """多模态融合层 / Multimodal fusion layer."""
    def __init__(self, embed_dim=768, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        
        if fusion_type == 'concat':
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        elif fusion_type == 'attention':
            self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text_features, vision_features):
        if self.fusion_type == 'concat':
            concat_features = torch.cat([text_features, vision_features], dim=-1)
            fused_features = self.fusion_proj(concat_features)
            fused_features = self.dropout(fused_features)
        elif self.fusion_type == 'attention':
            text_query = text_features.unsqueeze(1)
            vision_kv = vision_features.unsqueeze(1)
            attn_output, _ = self.cross_attention(text_query, vision_kv, vision_kv)
            fused_features = self.layer_norm(attn_output.squeeze(1) + text_features)
            fused_features = self.dropout(fused_features)
        
        return fused_features


class LinearProbingHead(nn.Module):
    """
    线性探测头 - 用于单任务分类
    编码器冻结，只训练这个头
    Linear probing head for single-task classification.
    Encoders are frozen; only this head is trained.
    """
    def __init__(self, input_dim=768, num_classes=2, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            # 简单线性层 / Simple linear layer.
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            # 带一层隐藏层的分类器 / Classifier with one hidden layer.
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
    
    def forward(self, features):
        return self.classifier(features)


class MultiTaskLinearHead(nn.Module):
    """
    多任务线性探测头
    每个任务一个独立的线性层
    Multi-task linear probing head.
    Each task has an independent linear layer.
    """
    def __init__(self, input_dim=768, task_configs=None):
        super().__init__()
        
        if task_configs is None:
            raise ValueError("task_configs must be provided")
        
        self.task_names = list(task_configs.keys())
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            num_classes = config.get('num_classes', 2)
            hidden_dim = config.get('hidden_dim', None)
            self.task_heads[task_name] = LinearProbingHead(input_dim, num_classes, hidden_dim)
    
    def forward(self, features):
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(features)
        return outputs


class BEiT3EDFoundationModel(nn.Module):
    """
    BEiT3-ED Foundation Model
    
    通用的急诊科医疗基础模型，支持：
    - 线性探测训练策略（编码器冻结）
    - 多种数据集和任务类型
    - 单模态和多模态输入
    - 零样本推理
    General-purpose medical foundation model for the Emergency Department,
    supporting:
    - Linear probing training strategy with frozen encoders
    - Multiple datasets and task types
    - Single-modal and multimodal inputs
    - Zero-shot inference
    """
    
    def __init__(
        self,
        vocab_size=64010,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        image_model_name='beit_base_patch16_224',
        pretrained_vision=False,
        dataset_name=None,
        task_type=None,
        config_path=None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dataset_name = dataset_name
        self.task_type = task_type
        
        # 加载配置 / Load configuration.
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None
        
        # 创建编码器（预训练，将被冻结） / Create pretrained encoders that will be frozen.
        timm_model = timm.create_model(
            image_model_name, 
            pretrained=pretrained_vision, 
            num_classes=0
        )
        
        self.text_encoder = TextEncoder(vocab_size, embed_dim, num_layers, num_heads)
        self.vision_encoder = VisionEncoder(timm_model)
        self.fusion_layer = MultimodalFusion(embed_dim, fusion_type='concat')
        self.image_mask_feature = nn.Parameter(torch.zeros(embed_dim))
        
        # 默认冻结编码器 / Freeze encoders by default.
        self.freeze_encoders()
        
        # 任务头（将在设置数据集时初始化） / Task head initialized when a dataset is set up.
        self.task_head = None
    
    def freeze_encoders(self):
        """冻结编码器参数（线性探测策略） / Freeze encoder parameters for linear probing."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.fusion_layer.parameters():
            param.requires_grad = False
        self.image_mask_feature.requires_grad = False
    
    def unfreeze_encoders(self):
        """解冻编码器（如需微调） / Unfreeze encoders when fine-tuning is needed."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.fusion_layer.parameters():
            param.requires_grad = True
        self.image_mask_feature.requires_grad = True
    
    def setup_dataset(self, dataset_name, task_type, config=None):
        """
        为特定数据集设置任务头
        Set up the task head for a specific dataset.
        
        Args:
            dataset_name: 数据集名称 / Dataset name.
            task_type: 任务类型 / Task type (early_triage/prognosis_prediction/full_process_decision).
            config: 数据集配置（可选） / Optional dataset configuration.
        """
        self.dataset_name = dataset_name
        self.task_type = task_type
        
        if config is None and self.config:
            config = self.config['datasets'][task_type][dataset_name]
        
        if config is None:
            raise ValueError(f"No configuration found for {dataset_name}")
        
        # 根据配置创建任务头 / Create task head from configuration.
        if 'tasks' in config:
            # 多任务 / Multi-task setup.
            task_configs = {}
            for task_name, task_config in config['tasks'].items():
                task_configs[task_name] = {
                    'num_classes': task_config['num_classes'],
                    'hidden_dim': None  # 单层线性探测头 / Single linear probing head.
                }
            self.task_head = MultiTaskLinearHead(self.embed_dim, task_configs)
        else:
            # 单任务 / Single-task setup.
            num_classes = config['num_classes']
            self.task_head = LinearProbingHead(self.embed_dim, num_classes, hidden_dim=None)
        
        self.modality = config.get('modality', 'text_only')
        self.is_multitask = 'tasks' in config
    
    @classmethod
    def from_pretrained(cls, model_path, dataset_name=None, task_type=None, config_path='config.json', **kwargs):
        """从预训练权重加载模型 / Load model from pretrained weights."""
        import os
        
        # 加载配置 / Load configuration.
        config_file = os.path.join(os.path.dirname(model_path), config_path) if os.path.isfile(model_path) else os.path.join(model_path, config_path)
        
        # 创建模型 / Create model.
        model = cls(config_path=config_file, **kwargs)
        
        # 加载权重 / Load weights.
        if os.path.isfile(model_path):
            checkpoint_path = model_path
        else:
            checkpoint_path = os.path.join(model_path, 'pytorch_model.bin')
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 只加载编码器权重（任务头会重新初始化） / Load encoder weights only; task heads are reinitialized.
        encoder_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('task_head') and not k.startswith('task_heads')}
        
        model.load_state_dict(encoder_state_dict, strict=False)
        print(f"✓ 编码器权重已从 {checkpoint_path} 加载")
        
        # 设置数据集和任务头 / Set dataset and task head.
        if dataset_name and task_type:
            model.setup_dataset(dataset_name, task_type)
            print(f"✓ 已配置数据集: {dataset_name} ({task_type})")
        
        return model
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        class_weights=None,
        return_dict=True
    ):
        """
        前向传播
        Forward pass.
        
        Args:
            input_ids: 文本输入 / Text input [batch_size, seq_len].
            attention_mask: 注意力掩码 / Attention mask [batch_size, seq_len].
            pixel_values: 图像输入 / Image input [batch_size, 3, H, W].
            labels: 标签 / Labels (single-task: [batch_size]; multi-task: dict).
            class_weights: 类别权重 / Optional class weights for imbalanced labels.
            return_dict: 是否返回字典格式 / Whether to return a dictionary.
        """
        # 编码 / Encode inputs.
        if self.modality == 'text_only':
            # 纯文本模态 / Text-only modality.
            features = self.text_encoder(input_ids, attention_mask)
        
        elif self.modality == 'multimodal':
            # 多模态 / Multimodal modality.
            text_features = self.text_encoder(input_ids, attention_mask)
            if pixel_values is None:
                batch_size = input_ids.shape[0]
                vision_features = self.image_mask_feature.unsqueeze(0).expand(batch_size, -1)
            else:
                vision_features = self.vision_encoder(pixel_values)
            features = self.fusion_layer(text_features, vision_features)
        
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
        
        # 分类 / Classify.
        if self.task_head is None:
            raise ValueError("Task head not initialized. Call setup_dataset() first.")
        
        logits = self.task_head(features)
        
        outputs = {
            'features': features,
            'logits': logits
        }
        
        # 计算损失 / Compute loss.
        if labels is not None:
            if self.is_multitask:
                # 多任务损失 / Multi-task loss.
                total_loss = 0.0
                losses = {}
                
                for task_name, task_logits in logits.items():
                    if task_name in labels:
                        task_labels = labels[task_name]
                        weight = class_weights.get(task_name) if isinstance(class_weights, dict) else None
                        loss = F.cross_entropy(task_logits, task_labels, weight=weight)
                        losses[f'{task_name}_loss'] = loss
                        total_loss += loss
                
                losses['total_loss'] = total_loss
                outputs['loss'] = total_loss
                outputs['losses'] = losses
            else:
                # 单任务损失 / Single-task loss.
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                outputs['loss'] = loss
        
        if return_dict:
            return outputs
        else:
            return tuple(outputs.values())
    
    def predict(self, input_ids, attention_mask=None, pixel_values=None):
        """推理接口 / Inference interface."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_dict=True
            )
        
        logits = outputs['logits']
        
        if self.is_multitask:
            # 多任务预测 / Multi-task prediction.
            predictions = {}
            probabilities = {}
            for task_name, task_logits in logits.items():
                probs = F.softmax(task_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                predictions[task_name] = preds
                probabilities[task_name] = probs
            return predictions, probabilities
        else:
            # 单任务预测 / Single-task prediction.
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            return preds, probs


# 简单的Tokenizer / Simple tokenizer.
class SimpleEDTokenizer:
    """简单的ED Foundation Model Tokenizer / Simple ED Foundation Model tokenizer."""
    
    def __init__(self, vocab_size=64010):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
        
    def __call__(self, text, max_length=512, padding='max_length', truncation=True, return_tensors='pt'):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        input_ids_list = []
        attention_mask_list = []
        
        for txt in texts:
            tokens = [self.cls_token_id]
            
            for char in txt:
                char_id = (ord(char) % (self.vocab_size - 10)) + 10
                tokens.append(char_id)
            
            tokens.append(self.sep_token_id)
            
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.sep_token_id]
            
            attention_mask = [1] * len(tokens)
            if padding == 'max_length':
                pad_length = max_length - len(tokens)
                tokens.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            input_ids_list.append(tokens)
            attention_mask_list.append(attention_mask)
        
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
            }
        else:
            return {
                'input_ids': input_ids_list,
                'attention_mask': attention_mask_list
            }


# 辅助函数 / Helper functions.
def get_dataset_config(dataset_name, task_type, config_path='config.json'):
    """获取数据集配置 / Get dataset configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['datasets'][task_type][dataset_name]


def list_supported_datasets(config_path='config.json'):
    """列出所有支持的数据集 / List all supported datasets."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("支持的数据集:")
    print("="*80)
    
    for task_type, datasets in config['datasets'].items():
        print(f"\n{task_type.upper().replace('_', ' ')}:")
        for dataset_name, dataset_config in datasets.items():
            modality = dataset_config.get('modality', 'unknown')
            training = "训练" if dataset_config.get('training_required') else "零样本"
            print(f"  - {dataset_name:25s} | {modality:15s} | {training}")
    
    print("="*80)
