#!/usr/bin/env python3
"""
BEiT3-ED Foundation Model 使用示例
"""

import torch
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer
from PIL import Image
import torchvision.transforms as transforms


def example_single_prediction():
    """单个样本预测示例"""
    print("="*80)
    print("示例1: 单个样本预测")
    print("="*80)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    model = BEiT3EDFoundationModel.from_pretrained('pytorch_model.bin')
    model.eval()
    print("✓ 模型加载完成")
    
    # 加载tokenizer
    print("\n[2/4] 加载tokenizer...")
    tokenizer = SimpleEDTokenizer(vocab_size=64010)
    print("✓ Tokenizer加载完成")
    
    # 准备文本输入
    print("\n[3/4] 准备输入数据...")
    text = "患者年龄65岁，男性，主诉胸痛3小时，血压150/90mmHg，心率95次/分，呼吸频率18次/分"
    text_inputs = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')
    
    # 准备图像输入（示例：创建随机图像，实际使用时应加载真实图像）
    # 实际使用：image = Image.open('path/to/chest_xray.jpg')
    image = Image.new('RGB', (224, 224), color='gray')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image).unsqueeze(0)
    
    print(f"✓ 文本输入形状: {text_inputs['input_ids'].shape}")
    print(f"✓ 图像输入形状: {pixel_values.shape}")
    
    # 模型推理
    print("\n[4/4] 执行推理...")
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=pixel_values
        )
    
    # 显示预测结果
    print("\n预测结果:")
    print("-" * 80)
    logits = outputs['logits']
    for task_name, task_logits in logits.items():
        probs = torch.softmax(task_logits, dim=-1)
        risk_prob = probs[0, 1].item()  # 阳性类别（高风险）概率
        
        # 风险等级
        if risk_prob < 0.3:
            risk_level = "低风险 🟢"
        elif risk_prob < 0.7:
            risk_level = "中风险 🟡"
        else:
            risk_level = "高风险 🔴"
        
        task_name_cn = {
            'mechanical_ventilation': '机械通气需求',
            'icu_stay': 'ICU转诊需求',
            'mortality_7d': '7天死亡风险',
            'mortality_28d': '28天死亡风险'
        }.get(task_name, task_name)
        
        print(f"{task_name_cn:15s}: {risk_prob:6.2%}  {risk_level}")
    
    print("-" * 80)
    print("\n✓ 推理完成！")


def example_batch_prediction():
    """批量预测示例"""
    print("\n" + "="*80)
    print("示例2: 批量预测")
    print("="*80)
    
    # 加载模型
    print("\n[1/3] 加载模型...")
    model = BEiT3EDFoundationModel.from_pretrained('pytorch_model.bin')
    model.eval()
    
    tokenizer = SimpleEDTokenizer(vocab_size=64010)
    print("✓ 模型和tokenizer加载完成")
    
    # 准备批量数据
    print("\n[2/3] 准备批量数据...")
    texts = [
        "患者1: 65岁男性，胸痛，血压150/90",
        "患者2: 45岁女性，呼吸困难，血氧饱和度92%",
        "患者3: 78岁男性，意识模糊，血糖15.6mmol/L"
    ]
    
    # Tokenize文本
    text_inputs = tokenizer(texts, max_length=512, padding='max_length', return_tensors='pt')
    
    # 创建批量图像（实际使用时应加载真实图像）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = [Image.new('RGB', (224, 224), color='gray') for _ in range(3)]
    pixel_values = torch.stack([transform(img) for img in images])
    
    print(f"✓ 批量大小: {len(texts)}")
    print(f"✓ 文本输入形状: {text_inputs['input_ids'].shape}")
    print(f"✓ 图像输入形状: {pixel_values.shape}")
    
    # 批量推理
    print("\n[3/3] 执行批量推理...")
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=pixel_values
        )
    
    # 显示结果
    print("\n批量预测结果:")
    print("="*80)
    logits = outputs['logits']
    
    for i in range(len(texts)):
        print(f"\n患者 {i+1}:")
        print("-" * 40)
        for task_name, task_logits in logits.items():
            probs = torch.softmax(task_logits, dim=-1)
            risk_prob = probs[i, 1].item()
            
            task_name_cn = {
                'mechanical_ventilation': '机械通气',
                'icu_stay': 'ICU转诊',
                'mortality_7d': '7天死亡',
                'mortality_28d': '28天死亡'
            }.get(task_name, task_name)
            
            print(f"  {task_name_cn:10s}: {risk_prob:6.2%}")
    
    print("\n" + "="*80)
    print("✓ 批量推理完成！")


def example_feature_extraction():
    """特征提取示例"""
    print("\n" + "="*80)
    print("示例3: 特征提取")
    print("="*80)
    
    # 加载模型
    print("\n[1/2] 加载模型...")
    model = BEiT3EDFoundationModel.from_pretrained('pytorch_model.bin')
    model.eval()
    
    tokenizer = SimpleEDTokenizer(vocab_size=64010)
    
    # 准备输入
    print("\n[2/2] 提取特征...")
    text = "患者年龄65岁，主诉胸痛"
    text_inputs = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')
    
    image = Image.new('RGB', (224, 224), color='gray')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image).unsqueeze(0)
    
    # 提取特征
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=pixel_values
        )
    
    # 显示特征
    print("\n提取的特征:")
    print("-" * 80)
    print(f"文本特征维度: {outputs['text_features'].shape}")
    print(f"视觉特征维度: {outputs['vision_features'].shape}")
    print(f"融合特征维度: {outputs['fused_features'].shape}")
    print("-" * 80)
    
    print("\n✓ 特征提取完成！")
    print("\n说明: 这些特征可以用于:")
    print("  - 下游任务的输入")
    print("  - 相似度计算")
    print("  - 聚类分析")
    print("  - 可视化分析")


def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "BEiT3-ED Foundation Model 使用示例" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    
    # 示例1: 单个样本预测
    example_single_prediction()
    
    # 示例2: 批量预测
    example_batch_prediction()
    
    # 示例3: 特征提取
    example_feature_extraction()
    
    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("="*80)
    print("\n更多信息请参考 README.md")
    print()


if __name__ == "__main__":
    main()

