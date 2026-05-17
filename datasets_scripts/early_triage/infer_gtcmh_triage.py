#!/usr/bin/env python3
"""
GTCMH-Triage 零样本推理脚本
广东省中医院急诊分诊 - 使用SYSMH-S-Triage训练的模型
"""

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer


class GTCMHTriageDataset(Dataset):
    """GTCMH-Triage 数据集 (零样本)"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        self.data['label'] = self.data['label'] - 1
        print(f"加载 {len(self.data)} 条测试数据 (GTCMH)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            str(row['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(row['label']), dtype=torch.long),
            'text': str(row['text'])
        }


def main():
    parser = argparse.ArgumentParser(description='GTCMH-Triage Zero-shot Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='SYSMH-S-Triage训练好的模型路径')
    parser.add_argument('--base_model_path', type=str, default='../../pytorch_model.bin')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output', type=str, default='gtcmh_predictions.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("="*80)
    print("GTCMH-Triage 零样本推理 (广东省中医院)")
    print("="*80 + "\n")
    
    # 加载模型
    model = BEiT3EDFoundationModel.from_pretrained(
        args.base_model_path,
        dataset_name='GTCMH-Triage',
        task_type='early_triage'
    )
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(args.device)
    model.eval()
    
    # 加载数据
    tokenizer = SimpleEDTokenizer()
    test_dataset = GTCMHTriageDataset(args.test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 推理
    all_preds, all_labels, all_probs, all_texts = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inferring'):
            outputs = model(
                input_ids=batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device)
            )
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
            all_probs.extend(probs.cpu().numpy())
            all_texts.extend(batch['text'])
    
    # 评估
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*80)
    print("零样本推理结果 (GTCMH)")
    print("="*80)
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n" + classification_report(all_labels, all_preds,
                                      target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                                      digits=4))
    
    # 保存结果
    pd.DataFrame({
        'text': all_texts,
        'true_label': [l+1 for l in all_labels],
        'pred_label': [p+1 for p in all_preds],
        'prob_level_1': [p[0] for p in all_probs],
        'prob_level_2': [p[1] for p in all_probs],
        'prob_level_3': [p[2] for p in all_probs],
        'prob_level_4': [p[3] for p in all_probs],
    }).to_csv(args.output, index=False, encoding='utf-8-sig')
    
    print(f"\n预测结果已保存到: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()


