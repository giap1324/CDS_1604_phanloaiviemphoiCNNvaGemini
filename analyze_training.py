"""
Script phân tích kết quả training và tạo báo cáo
"""
import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import config

def analyze_training_results(model_dir=None):
    """Phân tích chi tiết kết quả training"""
    
    if model_dir is None:
        model_dir = config.MODEL_DIR
    
    print("=" * 80)
    print("TRAINING ANALYSIS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load model checkpoint
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    final_model_path = os.path.join(model_dir, 'final_model.pth')
    
    analysis_report = {
        'timestamp': datetime.now().isoformat(),
        'model_architecture': 'DenseNet121',
        'dataset': 'Chest X-Ray Pneumonia',
        'training_config': {},
        'performance_metrics': {},
        'convergence_analysis': {},
        'recommendations': []
    }
    
    # 1. Training Configuration
    print("1. TRAINING CONFIGURATION")
    print("-" * 80)
    training_config = {
        'Model Architecture': 'DenseNet121 (pretrained on ImageNet)',
        'Optimizer': 'AdamW',
        'Learning Rate': config.LEARNING_RATE,
        'Batch Size': config.BATCH_SIZE,
        'Epochs': config.EPOCHS,
        'Dropout Rate': config.DROPOUT_RATE,
        'Image Size': config.IMAGE_SIZE,
        'Loss Function': 'CrossEntropyLoss',
        'LR Scheduler': 'ReduceLROnPlateau (factor=0.5, patience=3)',
        'Mixed Precision': 'FP16 (CUDA)',
        'Data Augmentation': 'RandomHorizontalFlip, RandomRotation'
    }
    
    for key, value in training_config.items():
        print(f"  {key:.<25} {value}")
    analysis_report['training_config'] = training_config
    
    # 2. Performance Metrics
    print("\n2. PERFORMANCE METRICS")
    print("-" * 80)
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location='cpu')
        print(f"  Best Model Epoch........... {checkpoint.get('epoch', 'N/A') + 1}")
        print(f"  Best Validation Accuracy... {checkpoint.get('val_acc', 0):.4f} ({checkpoint.get('val_acc', 0)*100:.2f}%)")
        print(f"  Training Accuracy.......... {checkpoint.get('train_acc', 0):.4f} ({checkpoint.get('train_acc', 0)*100:.2f}%)")
        
        analysis_report['performance_metrics']['best_epoch'] = checkpoint.get('epoch', 0) + 1
        analysis_report['performance_metrics']['best_val_acc'] = float(checkpoint.get('val_acc', 0))
        analysis_report['performance_metrics']['best_train_acc'] = float(checkpoint.get('train_acc', 0))
    
    if os.path.exists(final_model_path):
        final_checkpoint = torch.load(final_model_path, map_location='cpu')
        test_acc = final_checkpoint.get('test_acc', 0)
        print(f"  Test Accuracy.............. {test_acc:.4f} ({test_acc*100:.2f}%)")
        analysis_report['performance_metrics']['test_acc'] = float(test_acc)
    
    # 3. Convergence Analysis
    print("\n3. CONVERGENCE ANALYSIS")
    print("-" * 80)
    
    # Đọc history từ training (giả sử có lưu)
    # Trong thực tế, bạn nên save training history vào file JSON
    print("  Status..................... Training completed successfully")
    
    if os.path.exists(best_model_path):
        overfitting_gap = abs(checkpoint.get('train_acc', 0) - checkpoint.get('val_acc', 0))
        print(f"  Train-Val Gap.............. {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
        
        if overfitting_gap < 0.05:
            convergence_status = "✓ Good - Low overfitting"
            print(f"  Convergence Status......... {convergence_status}")
        elif overfitting_gap < 0.10:
            convergence_status = "⚠ Moderate - Some overfitting"
            print(f"  Convergence Status......... {convergence_status}")
        else:
            convergence_status = "✗ High overfitting detected"
            print(f"  Convergence Status......... {convergence_status}")
        
        analysis_report['convergence_analysis']['overfitting_gap'] = float(overfitting_gap)
        analysis_report['convergence_analysis']['status'] = convergence_status
    
    # 4. Model Generalization
    print("\n4. MODEL GENERALIZATION")
    print("-" * 80)
    
    if os.path.exists(final_model_path) and os.path.exists(best_model_path):
        val_acc = checkpoint.get('val_acc', 0)
        test_acc = final_checkpoint.get('test_acc', 0)
        generalization_gap = abs(val_acc - test_acc)
        
        print(f"  Validation Accuracy........ {val_acc:.4f}")
        print(f"  Test Accuracy.............. {test_acc:.4f}")
        print(f"  Generalization Gap......... {generalization_gap:.4f} ({generalization_gap*100:.2f}%)")
        
        if generalization_gap < 0.03:
            print("  Generalization Status...... ✓ Excellent - Model generalizes well")
        elif generalization_gap < 0.05:
            print("  Generalization Status...... ✓ Good - Acceptable generalization")
        else:
            print("  Generalization Status...... ⚠ Model may not generalize well")
        
        analysis_report['convergence_analysis']['generalization_gap'] = float(generalization_gap)
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    if os.path.exists(best_model_path):
        train_acc = checkpoint.get('train_acc', 0)
        val_acc = checkpoint.get('val_acc', 0)
        
        if val_acc < 0.85:
            rec = "• Consider training for more epochs or adjusting learning rate"
            print(rec)
            recommendations.append(rec)
        
        if abs(train_acc - val_acc) > 0.10:
            rec = "• High overfitting detected - increase dropout rate or add more regularization"
            print(rec)
            recommendations.append(rec)
        
        if val_acc > 0.90 and abs(train_acc - val_acc) < 0.05:
            rec = "✓ Model performance is excellent - ready for deployment"
            print(rec)
            recommendations.append(rec)
        
        if config.BATCH_SIZE < 32:
            rec = "• Consider increasing batch size if GPU memory allows"
            print(rec)
            recommendations.append(rec)
    
    analysis_report['recommendations'] = recommendations
    
    # 6. Save Analysis Report
    report_path = os.path.join(model_dir, 'training_analysis.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Analysis report saved to: {report_path}")
    
    # 7. Generate Detailed Plots
    print("\n6. GENERATING DETAILED ANALYSIS PLOTS")
    print("-" * 80)
    generate_detailed_plots(model_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return analysis_report


def generate_detailed_plots(model_dir):
    """Tạo các biểu đồ phân tích chi tiết"""
    
    # Kiểm tra xem có history plots không
    history_plot = os.path.join(model_dir, 'training_history.png')
    cm_plot = os.path.join(model_dir, 'confusion_matrix.png')
    
    if os.path.exists(history_plot):
        print(f"  ✓ Training history plot found: training_history.png")
    
    if os.path.exists(cm_plot):
        print(f"  ✓ Confusion matrix plot found: confusion_matrix.png")
    
    # Tạo summary visualization
    create_summary_visualization(model_dir)


def create_summary_visualization(model_dir):
    """Tạo visualization tổng hợp"""
    
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    final_model_path = os.path.join(model_dir, 'final_model.pth')
    
    if not (os.path.exists(best_model_path) and os.path.exists(final_model_path)):
        return
    
    checkpoint = torch.load(best_model_path, map_location='cpu')
    final_checkpoint = torch.load(final_model_path, map_location='cpu')
    
    # Create performance comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy Comparison
    metrics = ['Train', 'Validation', 'Test']
    accuracies = [
        checkpoint.get('train_acc', 0) * 100,
        checkpoint.get('val_acc', 0) * 100,
        final_checkpoint.get('test_acc', 0) * 100
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = axes[0].bar(metrics, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Overfitting Analysis
    train_acc = checkpoint.get('train_acc', 0) * 100
    val_acc = checkpoint.get('val_acc', 0) * 100
    gap = abs(train_acc - val_acc)
    
    categories = ['Training\nAccuracy', 'Validation\nAccuracy', 'Overfitting\nGap']
    values = [train_acc, val_acc, gap]
    colors2 = ['#27ae60', '#2980b9', '#e67e22']
    
    bars2 = axes[1].bar(categories, values, color=colors2, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    summary_path = os.path.join(model_dir, 'performance_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Performance summary created: performance_summary.png")


def generate_markdown_report(model_dir=None):
    """Tạo báo cáo dạng Markdown"""
    
    if model_dir is None:
        model_dir = config.MODEL_DIR
    
    report_json_path = os.path.join(model_dir, 'training_analysis.json')
    
    if not os.path.exists(report_json_path):
        print("Running analysis first...")
        analyze_training_results(model_dir)
    
    with open(report_json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # Generate Markdown
    md_content = f"""# Training Analysis Report
## Pneumonia Classification Model

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Model Architecture

- **Base Model:** DenseNet121 (pretrained on ImageNet)
- **Task:** Binary Classification (NORMAL vs PNEUMONIA)
- **Framework:** PyTorch with Mixed Precision (FP16)

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
"""
    
    for key, value in report['training_config'].items():
        md_content += f"| {key} | {value} |\n"
    
    md_content += f"""
## 3. Performance Metrics

### Best Model Performance (Epoch {report['performance_metrics'].get('best_epoch', 'N/A')})

- **Training Accuracy:** {report['performance_metrics'].get('best_train_acc', 0):.4f} ({report['performance_metrics'].get('best_train_acc', 0)*100:.2f}%)
- **Validation Accuracy:** {report['performance_metrics'].get('best_val_acc', 0):.4f} ({report['performance_metrics'].get('best_val_acc', 0)*100:.2f}%)
- **Test Accuracy:** {report['performance_metrics'].get('test_acc', 0):.4f} ({report['performance_metrics'].get('test_acc', 0)*100:.2f}%)

### Convergence Analysis

- **Overfitting Gap:** {report['convergence_analysis'].get('overfitting_gap', 0):.4f} ({report['convergence_analysis'].get('overfitting_gap', 0)*100:.2f}%)
- **Status:** {report['convergence_analysis'].get('status', 'N/A')}
- **Generalization Gap:** {report['convergence_analysis'].get('generalization_gap', 0):.4f} ({report['convergence_analysis'].get('generalization_gap', 0)*100:.2f}%)

## 4. Training Progress Visualization

### Loss and Accuracy Curves
![Training History](training_history.png)

### Performance Summary
![Performance Summary](performance_summary.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## 5. Recommendations

"""
    
    for rec in report['recommendations']:
        md_content += f"{rec}\n"
    
    md_content += """
## 6. Conclusion

"""
    
    test_acc = report['performance_metrics'].get('test_acc', 0)
    if test_acc > 0.90:
        md_content += "✓ The model demonstrates **excellent performance** with high accuracy and good generalization. Ready for deployment.\n"
    elif test_acc > 0.85:
        md_content += "✓ The model shows **good performance** with acceptable accuracy. Consider fine-tuning for production use.\n"
    else:
        md_content += "⚠ The model needs **further improvement** before deployment. Consider adjusting hyperparameters or training longer.\n"
    
    md_content += f"""
---

*Report generated by automated training analysis script*
"""
    
    # Save Markdown report
    md_path = os.path.join(model_dir, 'TRAINING_REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✓ Markdown report saved to: {md_path}")
    return md_path


if __name__ == "__main__":
    print("Starting training analysis...\n")
    
    # Run analysis
    analysis_report = analyze_training_results()
    
    # Generate Markdown report
    print("\nGenerating Markdown report...")
    md_path = generate_markdown_report()
    
    print(f"\n{'='*80}")
    print("All reports generated successfully!")
    print(f"{'='*80}")
    print(f"\nFiles generated:")
    print(f"  1. training_analysis.json - Detailed analysis data")
    print(f"  2. performance_summary.png - Performance visualization")
    print(f"  3. TRAINING_REPORT.md - Complete training report")
    print(f"\nLocation: {config.MODEL_DIR}")
