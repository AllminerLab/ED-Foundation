#!/usr/bin/env python3
"""
SYSMH-N-Triage zero-shot inference script.
Uses a model trained on SYSMH-S-Triage for zero-shot inference.
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
    """SYSMH-N-Triage dataset (zero-shot)."""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        self.data['label'] = self.data['label'] - 1  # 1-4 -> 0-3
        print(f"Loaded {len(self.data)} test records")
    
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
                       help='Path to the trained SYSMH-S-Triage model')
    parser.add_argument('--base_model_path', type=str, default='../../pytorch_model.bin',
                       help='Base model path')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Test data path')
    parser.add_argument('--output', type=str, default='sysmh_n_predictions.csv',
                       help='Output prediction file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("="*80)
    print("SYSMH-N-Triage zero-shot inference using the SYSMH-S-Triage model")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print("="*80 + "\n")
    
    # Load model.
    print("[1/3] Loading model...")
    model = BEiT3EDFoundationModel.from_pretrained(
        args.base_model_path,
        dataset_name='SYSMH-N-Triage',  # Use the N-Triage configuration with the same 4 classes.
        task_type='early_triage'
    )
    
    # Load weights trained on S-Triage.
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(args.device)
    model.eval()
    print("Model loaded\n")
    
    # Load data.
    print("[2/3] Loading test data...")
    tokenizer = SimpleEDTokenizer()
    test_dataset = SYSMHNTriageDataset(args.test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Data loaded\n")
    
    # Zero-shot inference.
    print("[3/3] Running zero-shot inference...")
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
    
    # Evaluate.
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*80)
    print("Zero-shot inference results")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                                digits=4))
    
    print("\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # Save predictions.
    results_df = pd.DataFrame({
        'text': all_texts,
        'true_label': [l+1 for l in all_labels],  # Convert back to 1-4.
        'pred_label': [p+1 for p in all_preds],
        'prob_level_1': [p[0] for p in all_probs],
        'prob_level_2': [p[1] for p in all_probs],
        'prob_level_3': [p[2] for p in all_probs],
        'prob_level_4': [p[3] for p in all_probs],
    })
    
    results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nPredictions saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
