#!/usr/bin/env python3
"""
SYSMH-N-Triage 零样本推理脚本
使用SYSMH-S-Triage训练的模型进行零样本推理
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


class SYSMHNTriageDataset(Dataset):
    """SYSMH-N-Triage 数据集 (零样本)"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        self.data['label'] = self.data['label'] - 1  # 1-4 -> 0-3
        print(f"加载 {len(self.data)} 条测试数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        
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
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def main():
    parser = argparse.ArgumentParser(description='SYSMH-N-Triage Zero-shot Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='SYSMH-S-Triage训练好的模型路径')
    parser.add_argument('--base_model_path', type=str, default='../../pytorch_model.bin',
                       help='基础模型路径')
    parser.add_argument('--test_data', type=str, required=True,
                       help='测试数据路径')
    parser.add_argument('--output', type=str, default='sysmh_n_predictions.csv',
                       help='输出预测结果文件')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("="*80)
    print("SYSMH-N-Triage 零样本推理 (使用SYSMH-S-Triage模型)")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"测试数据: {args.test_data}")
    print("="*80 + "\n")
    
    # 加载模型
    print("[1/3] 加载模型...")
    model = BEiT3EDFoundationModel.from_pretrained(
        args.base_model_path,
        dataset_name='SYSMH-N-Triage',  # 使用N-Triage配置（相同的4分类）
        task_type='early_triage'
    )
    
    # 加载S-Triage训练好的权重
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(args.device)
    model.eval()
    print("✓ 模型加载完成\n")
    
    # 加载数据
    print("[2/3] 加载测试数据...")
    tokenizer = SimpleEDTokenizer()
    test_dataset = SYSMHNTriageDataset(args.test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("✓ 数据加载完成\n")
    
    # 零样本推理
    print("[3/3] 执行零样本推理...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inferring'):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_texts.extend(batch['text'])
    
    # 评估
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*80)
    print("零样本推理结果")
    print("="*80)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                                digits=4))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'text': all_texts,
        'true_label': [l+1 for l in all_labels],  # 转回1-4
        'pred_label': [p+1 for p in all_preds],
        'prob_level_1': [p[0] for p in all_probs],
        'prob_level_2': [p[1] for p in all_probs],
        'prob_level_3': [p[2] for p in all_probs],
        'prob_level_4': [p[3] for p in all_probs],
    })
    
    results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()

