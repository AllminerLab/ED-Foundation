#!/usr/bin/env python3
"""
BEiT3-ED Foundation Model usage examples.
"""

import torch
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer
from PIL import Image
import torchvision.transforms as transforms


def example_single_prediction():
    """Single-sample prediction example."""
    print("="*80)
    print("Example 1: single-sample prediction")
    print("="*80)
    
    # Load model.
    print("\n[1/4] Loading model...")
    model = BEiT3EDFoundationModel.from_pretrained('pytorch_model.bin')
    model.eval()
    print("Model loaded")
    
    # Load tokenizer.
    print("\n[2/4] Loading tokenizer...")
    tokenizer = SimpleEDTokenizer(vocab_size=64010)
    print("Tokenizer loaded")
    
    # Prepare text input.
    print("\n[3/4] Preparing inputs...")
    text = "65-year-old male patient with chest pain for 3 hours, blood pressure 150/90 mmHg, heart rate 95 bpm, respiratory rate 18/min"
    text_inputs = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')
    
    # Prepare image input; this example creates a placeholder image.
    # In real use: image = Image.open('path/to/chest_xray.jpg')
    image = Image.new('RGB', (224, 224), color='gray')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image).unsqueeze(0)
    
    print(f"Text input shape: {text_inputs['input_ids'].shape}")
    print(f"Image input shape: {pixel_values.shape}")
    
    # Run model inference.
    print("\n[4/4] Running inference...")
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=pixel_values
        )
    
    # Display predictions.
    print("\nPrediction results:")
    print("-" * 80)
    logits = outputs['logits']
    for task_name, task_logits in logits.items():
        probs = torch.softmax(task_logits, dim=-1)
        risk_prob = probs[0, 1].item()  # Positive-class risk probability.
        
        # Risk level.
        if risk_prob < 0.3:
            risk_level = "Low risk"
        elif risk_prob < 0.7:
            risk_level = "Medium risk"
        else:
            risk_level = "High risk"
        
        task_display_name = {
            'mechanical_ventilation': 'Mechanical ventilation',
            'icu_stay': 'ICU stay',
            'mortality_7d': '7-day mortality',
            'mortality_28d': '28-day mortality'
        }.get(task_name, task_name)
        
        print(f"{task_display_name:25s}: {risk_prob:6.2%}  {risk_level}")
    
    print("-" * 80)
    print("\nInference complete.")


def example_batch_prediction():
    """Batch prediction example."""
    print("\n" + "="*80)
    print("Example 2: batch prediction")
    print("="*80)
    
    # Load model.
    print("\n[1/3] Loading model...")
    model = BEiT3EDFoundationModel.from_pretrained('pytorch_model.bin')
    model.eval()
    
    tokenizer = SimpleEDTokenizer(vocab_size=64010)
    print("Model and tokenizer loaded")
    
    # Prepare batch data.
    print("\n[2/3] Preparing batch data...")
    texts = [
        "Patient 1: 65-year-old male, chest pain, blood pressure 150/90",
        "Patient 2: 45-year-old female, dyspnea, oxygen saturation 92%",
        "Patient 3: 78-year-old male, altered consciousness, glucose 15.6 mmol/L"
    ]
    
    # Tokenize text.
    text_inputs = tokenizer(texts, max_length=512, padding='max_length', return_tensors='pt')
    
    # Create placeholder batch images; real use should load actual images.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = [Image.new('RGB', (224, 224), color='gray') for _ in range(3)]
    pixel_values = torch.stack([transform(img) for img in images])
    
    print(f"Batch size: {len(texts)}")
    print(f"Text input shape: {text_inputs['input_ids'].shape}")
    print(f"Image input shape: {pixel_values.shape}")
    
    # Batch inference.
    print("\n[3/3] Running batch inference...")
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=pixel_values
        )
    
    # Display results.
    print("\nBatch prediction results:")
    print("="*80)
    logits = outputs['logits']
    
    for i in range(len(texts)):
        print(f"\nPatient {i+1}:")
        print("-" * 40)
        for task_name, task_logits in logits.items():
            probs = torch.softmax(task_logits, dim=-1)
            risk_prob = probs[i, 1].item()
            
            task_display_name = {
                'mechanical_ventilation': 'Mechanical ventilation',
                'icu_stay': 'ICU stay',
                'mortality_7d': '7-day mortality',
                'mortality_28d': '28-day mortality'
            }.get(task_name, task_name)
            
            print(f"  {task_display_name:25s}: {risk_prob:6.2%}")
    
    print("\n" + "="*80)
    print("Batch inference complete.")


def example_feature_extraction():
    """Feature extraction example."""
    print("\n" + "="*80)
    print("Example 3: feature extraction")
    print("="*80)
    
    # Load model.
    print("\n[1/2] Loading model...")
    model = BEiT3EDFoundationModel.from_pretrained('pytorch_model.bin')
    model.eval()
    
    tokenizer = SimpleEDTokenizer(vocab_size=64010)
    
    # Prepare input.
    print("\n[2/2] Extracting features...")
    text = "65-year-old patient with chest pain"
    text_inputs = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')
    
    image = Image.new('RGB', (224, 224), color='gray')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image).unsqueeze(0)
    
    # Extract features.
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=pixel_values
        )
    
    # Display features.
    print("\nExtracted features:")
    print("-" * 80)
    print(f"Feature shape: {outputs['features'].shape}")
    print("-" * 80)
    
    print("\nFeature extraction complete.")
    print("\nThese features can be used for:")
    print("  - Downstream task inputs")
    print("  - Similarity calculation")
    print("  - Clustering analysis")
    print("  - Visualization")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "BEiT3-ED Foundation Model Examples" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    # Example 1: single-sample prediction.
    example_single_prediction()
    
    # Example 2: batch prediction.
    example_batch_prediction()
    
    # Example 3: feature extraction.
    example_feature_extraction()
    
    print("\n" + "="*80)
    print("All examples completed.")
    print("="*80)
    print("\nSee README.md for more information.")
    print()


if __name__ == "__main__":
    main()
