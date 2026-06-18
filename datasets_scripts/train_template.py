#!/usr/bin/env python3
"""
BEiT3-ED Foundation Model - generic training script template
Linear probing training with frozen encoders.
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

# Add the parent directory to the Python path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class EDDataset(Dataset):
    """Base ED dataset class; implement dataset-specific logic in subclasses."""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, data_path):
        """Load data in a subclass."""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return one sample in a subclass."""
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
    Linear probing training.
    
    Args:
        model: BEiT3EDFoundationModel
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device
        save_dir: Save directory
    """
    
    # Ensure encoders are frozen.
    model.freeze_encoders()
    model = model.to(device)
    
    # Optimize only task-head parameters.
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
    print("Starting linear probing training")
    print("="*80)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        # Training phase.
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            # Move tensors to device.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            # Forward pass.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics.
            train_loss += loss.item()
            
            if isinstance(logits, dict):
                # Multi-task: calculate accuracy for the first task.
                task_name = list(logits.keys())[0]
                preds = torch.argmax(logits[task_name], dim=-1)
                train_correct += (preds == labels[task_name]).sum().item()
            else:
                # Single-task.
                preds = torch.argmax(logits, dim=-1)
                train_correct += (preds == labels).sum().item()
            
            train_total += labels.size(0) if not isinstance(labels, dict) else list(labels.values())[0].size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase.
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
        
        # Update learning rate.
        scheduler.step()
        
        # Print results.
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save the best model.
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
            print(f'  Saved best model: {save_path} (Val Acc: {val_acc:.2f}%)')
    
    print("\n" + "="*80)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    print("="*80 + "\n")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='BEiT3-ED Linear Probing Training')
    
    parser.add_argument('--model_path', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--task_type', type=str, required=True,
                       choices=['early_triage', 'prognosis_prediction', 'full_process_decision'],
                       help='Task type')
    parser.add_argument('--train_data', type=str, required=True, help='Training data path')
    parser.add_argument('--val_data', type=str, required=True, help='Validation data path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load model.
    print("Loading pretrained model...")
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path,
        dataset_name=args.dataset_name,
        task_type=args.task_type
    )
    
    # Create datasets; implement dataset-specific loading as needed.
    tokenizer = SimpleEDTokenizer()
    
    print("Loading data...")
    # train_dataset = YourDataset(args.train_data, tokenizer)
    # val_dataset = YourDataset(args.val_data, tokenizer)
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train.
    # train_linear_probing(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=args.num_epochs,
    #     learning_rate=args.learning_rate,
    #     device=args.device,
    #     save_dir=args.save_dir
    # )
    
    print("Please implement dataset-specific data loading.")


if __name__ == '__main__':
    main()
