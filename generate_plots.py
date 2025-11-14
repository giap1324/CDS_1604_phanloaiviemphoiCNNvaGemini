"""
Script tạo các biểu đồ phân tích và visualization cho model
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import config
from train_model import PneumoniaDataset, PneumoniaClassifier

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_model(model_path, device='cuda'):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    model = PneumoniaClassifier(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=False
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model

def get_predictions(model, dataloader, device):
    """Lấy predictions và labels từ model"""
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix_detailed(y_true, y_pred, class_names, save_path):
    """Vẽ confusion matrix chi tiết với percentages"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute numbers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Plot 2: Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax2)
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Detailed confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_probs, class_names, save_path):
    """Vẽ ROC curve cho binary classification"""
    # ROC curve cho class PNEUMONIA (class 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Pneumonia Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {save_path}")
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_probs, save_path):
    """Vẽ Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs[:, 1])
    avg_precision = average_precision_score(y_true, y_probs[:, 1])
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curve saved to {save_path}")
    
    return avg_precision

def plot_classification_metrics(y_true, y_pred, class_names, save_path):
    """Vẽ biểu đồ các metrics: Precision, Recall, F1-Score"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate metrics cho mỗi class
    metrics_data = {
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for i in range(len(class_names)):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        metrics_data['Precision'].append(precision_score(y_true_binary, y_pred_binary))
        metrics_data['Recall'].append(recall_score(y_true_binary, y_pred_binary))
        metrics_data['F1-Score'].append(f1_score(y_true_binary, y_pred_binary))
    
    # Create plot
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, metrics_data['Precision'], width, 
                   label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, metrics_data['Recall'], width, 
                   label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, metrics_data['F1-Score'], width, 
                   label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Classification metrics saved to {save_path}")

def plot_sample_predictions(model, dataset, device, num_samples=8, save_path=None):
    """Hiển thị sample predictions với ảnh gốc"""
    model.eval()
    
    # Random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    class_names = ['NORMAL', 'PNEUMONIA']
    
    for idx, sample_idx in enumerate(indices):
        image, label = dataset[sample_idx]
        
        # Get prediction
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = prob[0, pred].item()
        
        # Denormalize image for display
        img_display = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        
        # Plot
        axes[idx].imshow(img_display)
        axes[idx].axis('off')
        
        # Title with prediction
        true_label = class_names[label]
        pred_label = class_names[pred]
        color = 'green' if pred == label else 'red'
        
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}'
        axes[idx].set_title(title, fontsize=10, fontweight='bold', color=color)
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Sample predictions saved to {save_path}")
    else:
        plt.show()

def plot_error_analysis(y_true, y_pred, y_probs, class_names, save_path):
    """Phân tích errors"""
    # Tìm false positives và false negatives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Error counts
    error_counts = [len(fp_indices), len(fn_indices)]
    error_labels = ['False Positives\n(Normal → Pneumonia)', 
                   'False Negatives\n(Pneumonia → Normal)']
    colors = ['#e74c3c', '#f39c12']
    
    bars = ax1.bar(error_labels, error_counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 2: Confidence distribution of errors
    if len(fp_indices) > 0:
        fp_confidences = y_probs[fp_indices, 1]
        ax2.hist(fp_confidences, bins=20, alpha=0.6, color='#e74c3c', 
                label='False Positives', edgecolor='black')
    
    if len(fn_indices) > 0:
        fn_confidences = y_probs[fn_indices, 0]
        ax2.hist(fn_confidences, bins=20, alpha=0.6, color='#f39c12', 
                label='False Negatives', edgecolor='black')
    
    ax2.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Error Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error analysis saved to {save_path}")

def main():
    """Main function"""
    print("=" * 80)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first!")
        return
    
    model = load_model(model_path, device)
    
    # Load test data
    print("\nLoading test dataset...")
    test_dir = os.path.join(config.PROCESSED_DATA_DIR, 'test')
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = PneumoniaDataset(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Get predictions
    print("Getting predictions...")
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)
    
    class_names = ['NORMAL', 'PNEUMONIA']
    plots_dir = config.MODEL_DIR
    
    print(f"\nGenerating plots in {plots_dir}...\n")
    
    # 1. Detailed Confusion Matrix
    plot_confusion_matrix_detailed(
        y_true, y_pred, class_names,
        os.path.join(plots_dir, 'confusion_matrix_detailed.png')
    )
    
    # 2. ROC Curve
    roc_auc = plot_roc_curve(
        y_true, y_probs, class_names,
        os.path.join(plots_dir, 'roc_curve.png')
    )
    
    # 3. Precision-Recall Curve
    avg_precision = plot_precision_recall_curve(
        y_true, y_probs,
        os.path.join(plots_dir, 'precision_recall_curve.png')
    )
    
    # 4. Classification Metrics
    plot_classification_metrics(
        y_true, y_pred, class_names,
        os.path.join(plots_dir, 'classification_metrics.png')
    )
    
    # 5. Error Analysis
    plot_error_analysis(
        y_true, y_pred, y_probs, class_names,
        os.path.join(plots_dir, 'error_analysis.png')
    )
    
    # 6. Sample Predictions
    plot_sample_predictions(
        model, test_dataset, device, num_samples=8,
        save_path=os.path.join(plots_dir, 'sample_predictions.png')
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated plots:")
    print(f"  1. confusion_matrix_detailed.png - Detailed confusion matrix")
    print(f"  2. roc_curve.png - ROC curve (AUC = {roc_auc:.3f})")
    print(f"  3. precision_recall_curve.png - Precision-Recall curve (AP = {avg_precision:.3f})")
    print(f"  4. classification_metrics.png - Precision, Recall, F1-Score")
    print(f"  5. error_analysis.png - Error distribution and confidence")
    print(f"  6. sample_predictions.png - Sample predictions with images")
    print(f"\nLocation: {plots_dir}")

if __name__ == "__main__":
    main()
