#!/usr/bin/env python3
"""
GTCMH-Triage zero-shot inference script.
Guangdong Provincial TCM Hospital ED triage using a model trained on SYSMH-S-Triage.
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
    """GTCMH-Triage dataset (zero-shot)."""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        self.data['label'] = self.data['label'] - 1
        print(f"Loaded {len(self.data)} test records (GTCMH)")
    
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
                       help='Path to the trained SYSMH-S-Triage model')
    parser.add_argument('--base_model_path', type=str, default='../../pytorch_model.bin')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output', type=str, default='gtcmh_predictions.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("="*80)
    print("GTCMH-Triage zero-shot inference (Guangdong Provincial TCM Hospital)")
    print("="*80 + "\n")
    
    # Load model.
    model = BEiT3EDFoundationModel.from_pretrained(
        args.base_model_path,
        dataset_name='GTCMH-Triage',
        task_type='early_triage'
    )
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(args.device)
    model.eval()
    
    # Load data.
    tokenizer = SimpleEDTokenizer()
    test_dataset = GTCMHTriageDataset(args.test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Inference.
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
    
    # Evaluate.
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*80)
    print("Zero-shot inference results (GTCMH)")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\n" + classification_report(all_labels, all_preds,
                                      target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                                      digits=4))
    
    # Save results.
    pd.DataFrame({
        'text': all_texts,
        'true_label': [l+1 for l in all_labels],
        'pred_label': [p+1 for p in all_preds],
        'prob_level_1': [p[0] for p in all_probs],
        'prob_level_2': [p[1] for p in all_probs],
        'prob_level_3': [p[2] for p in all_probs],
        'prob_level_4': [p[3] for p in all_probs],
    }).to_csv(args.output, index=False, encoding='utf-8-sig')
    
    print(f"\nPredictions saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()

