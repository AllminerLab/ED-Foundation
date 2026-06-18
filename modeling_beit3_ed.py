#!/usr/bin/env python3
"""
BEiT3-ED Foundation Model - Emergency Department Foundation Model
General-purpose foundation model for the Emergency Department.

Supports three major task categories:
1. Early ED Triage
2. Prognosis Prediction
3. Full-Process ED Decision

Training strategy: Linear Probing - freeze encoders and train only task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
import timm
import json


class TextEncoder(nn.Module):
    """Text encoder based on Transformer."""
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
    """Vision encoder based on BEiT."""
    def __init__(self, timm_model):
        super().__init__()
        self.backbone = timm_model
        
    def forward(self, images):
        vision_features = self.backbone(images)
        return vision_features


class MultimodalFusion(nn.Module):
    """Multimodal fusion layer."""
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
    Linear probing head for single-task classification.
    Encoders are frozen; only this head is trained.
    """
    def __init__(self, input_dim=768, num_classes=2, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            # Simple linear layer.
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            # Classifier with one hidden layer.
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
        
        # Load configuration.
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None
        
        # Create pretrained encoders that will be frozen.
        timm_model = timm.create_model(
            image_model_name, 
            pretrained=pretrained_vision, 
            num_classes=0
        )
        
        self.text_encoder = TextEncoder(vocab_size, embed_dim, num_layers, num_heads)
        self.vision_encoder = VisionEncoder(timm_model)
        self.fusion_layer = MultimodalFusion(embed_dim, fusion_type='concat')
        self.image_mask_feature = nn.Parameter(torch.zeros(embed_dim))
        
        # Freeze encoders by default.
        self.freeze_encoders()
        
        # Task head initialized when a dataset is set up.
        self.task_head = None
    
    def freeze_encoders(self):
        """Freeze encoder parameters for linear probing."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.fusion_layer.parameters():
            param.requires_grad = False
        self.image_mask_feature.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze encoders when fine-tuning is needed."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.fusion_layer.parameters():
            param.requires_grad = True
        self.image_mask_feature.requires_grad = True
    
    def setup_dataset(self, dataset_name, task_type, config=None):
        """
        Set up the task head for a specific dataset.
        
        Args:
            dataset_name: Dataset name.
            task_type: Task type (early_triage/prognosis_prediction/full_process_decision).
            config: Optional dataset configuration.
        """
        self.dataset_name = dataset_name
        self.task_type = task_type
        
        if config is None and self.config:
            config = self.config['datasets'][task_type][dataset_name]
        
        if config is None:
            raise ValueError(f"No configuration found for {dataset_name}")
        
        # Create task head from configuration.
        if 'tasks' in config:
            # Multi-task setup.
            task_configs = {}
            for task_name, task_config in config['tasks'].items():
                task_configs[task_name] = {
                    'num_classes': task_config['num_classes'],
                    'hidden_dim': None  # Single linear probing head.
                }
            self.task_head = MultiTaskLinearHead(self.embed_dim, task_configs)
        else:
            # Single-task setup.
            num_classes = config['num_classes']
            self.task_head = LinearProbingHead(self.embed_dim, num_classes, hidden_dim=None)
        
        self.modality = config.get('modality', 'text_only')
        self.is_multitask = 'tasks' in config
    
    @classmethod
    def from_pretrained(cls, model_path, dataset_name=None, task_type=None, config_path='config.json', **kwargs):
        """Load model from pretrained weights."""
        import os
        
        # Load configuration.
        config_file = os.path.join(os.path.dirname(model_path), config_path) if os.path.isfile(model_path) else os.path.join(model_path, config_path)
        
        # Create model.
        model = cls(config_path=config_file, **kwargs)
        
        # Load weights.
        if os.path.isfile(model_path):
            checkpoint_path = model_path
        else:
            checkpoint_path = os.path.join(model_path, 'pytorch_model.bin')
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load encoder weights only; task heads are reinitialized.
        encoder_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('task_head') and not k.startswith('task_heads')}
        
        model.load_state_dict(encoder_state_dict, strict=False)
        print(f"Encoder weights loaded from {checkpoint_path}")
        
        # Set dataset and task head.
        if dataset_name and task_type:
            model.setup_dataset(dataset_name, task_type)
            print(f"Configured dataset: {dataset_name} ({task_type})")
        
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
        Forward pass.
        
        Args:
            input_ids: Text input [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            pixel_values: Image input [batch_size, 3, H, W].
            labels: Labels (single-task: [batch_size]; multi-task: dict).
            class_weights: Optional class weights for imbalanced labels.
            return_dict: Whether to return a dictionary.
        """
        # Encode inputs.
        if self.modality == 'text_only':
            # Text-only modality.
            features = self.text_encoder(input_ids, attention_mask)
        
        elif self.modality == 'multimodal':
            # Multimodal modality.
            text_features = self.text_encoder(input_ids, attention_mask)
            if pixel_values is None:
                batch_size = input_ids.shape[0]
                vision_features = self.image_mask_feature.unsqueeze(0).expand(batch_size, -1)
            else:
                vision_features = self.vision_encoder(pixel_values)
            features = self.fusion_layer(text_features, vision_features)
        
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
        
        # Classify.
        if self.task_head is None:
            raise ValueError("Task head not initialized. Call setup_dataset() first.")
        
        logits = self.task_head(features)
        
        outputs = {
            'features': features,
            'logits': logits
        }
        
        # Compute loss.
        if labels is not None:
            if self.is_multitask:
                # Multi-task loss.
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
                # Single-task loss.
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                outputs['loss'] = loss
        
        if return_dict:
            return outputs
        else:
            return tuple(outputs.values())
    
    def predict(self, input_ids, attention_mask=None, pixel_values=None):
        """Inference interface."""
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
            # Multi-task prediction.
            predictions = {}
            probabilities = {}
            for task_name, task_logits in logits.items():
                probs = F.softmax(task_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                predictions[task_name] = preds
                probabilities[task_name] = probs
            return predictions, probabilities
        else:
            # Single-task prediction.
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            return preds, probs


# Simple tokenizer.
class SimpleEDTokenizer:
    """Simple ED Foundation Model tokenizer."""
    
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


# Helper functions.
def get_dataset_config(dataset_name, task_type, config_path='config.json'):
    """Get dataset configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['datasets'][task_type][dataset_name]


def list_supported_datasets(config_path='config.json'):
    """List all supported datasets."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Supported datasets:")
    print("="*80)
    
    for task_type, datasets in config['datasets'].items():
        print(f"\n{task_type.upper().replace('_', ' ')}:")
        for dataset_name, dataset_config in datasets.items():
            modality = dataset_config.get('modality', 'unknown')
            training = "train" if dataset_config.get('training_required') else "zero-shot"
            print(f"  - {dataset_name:25s} | {modality:15s} | {training}")
    
    print("="*80)
