#!/usr/bin/env python3
"""
MIMIC-IV-ED-AMI 训练脚本
MIMIC-IV急诊AMI患者院内死亡预测 - 二分类任务
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class MIMICAMIDataset(Dataset):
    """MIMIC-IV-ED-AMI 数据集 - 死亡预测"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        # label: 0=Survival, 1=Mortality
        print(f"加载 {len(self.data)} 条数据")
        print(f"类别分布: Survival={len(self.data[self.data['label']==0])}, Mortality={len(self.data[self.data['label']==1])}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(str(row['text']), max_length=self.max_length, 
                                  padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(row['label']), dtype=torch.long)
        }


def train_and_eval(model, train_loader, val_loader, optimizer, device, num_epochs, save_dir):
    best_auc = 0.0
    history = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss, train_preds, train_labels, train_probs = 0.0, [], [], []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Train'):
            outputs = model(input_ids=batch['input_ids'].to(device),
                          attention_mask=batch['attention_mask'].to(device),
                          labels=batch['labels'].to(device))
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            logits = outputs['logits']
            train_probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            train_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            train_labels.extend(batch['labels'].numpy())
        
        # Eval
        model.eval()
        val_loss, val_preds, val_labels, val_probs = 0.0, [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1} Val'):
                outputs = model(input_ids=batch['input_ids'].to(device),
                              attention_mask=batch['attention_mask'].to(device),
                              labels=batch['labels'].to(device))
                val_loss += outputs['loss'].item()
                logits = outputs['logits']
                val_probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_labels.extend(batch['labels'].numpy())
        
        train_auc = roc_auc_score(train_labels, train_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        
        print(f"\nEpoch {epoch+1}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'model_state_dict': model.state_dict(), 'val_auc': val_auc},
                      os.path.join(save_dir, 'best_model.pt'))
    
    return best_auc


def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV-ED-AMI Training')
    parser.add_argument('--model_path', type=str, default='../../pytorch_model.bin')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mimic_ami')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("MIMIC-IV-ED-AMI 训练 - 院内死亡预测")
    print("="*80 + "\n")
    
    model = BEiT3EDFoundationModel.from_pretrained(
        args.model_path, dataset_name='MIMIC-IV-ED-AMI', task_type='prognosis_prediction')
    model.freeze_encoders()
    model = model.to(args.device)
    
    tokenizer = SimpleEDTokenizer()
    train_loader = DataLoader(MIMICAMIDataset(args.train_data, tokenizer), 
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(MIMICAMIDataset(args.val_data, tokenizer),
                           batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    optimizer = optim.AdamW(model.task_head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    best_auc = train_and_eval(model, train_loader, val_loader, optimizer, args.device, args.num_epochs, args.save_dir)
    print(f"\n训练完成！最佳AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()


