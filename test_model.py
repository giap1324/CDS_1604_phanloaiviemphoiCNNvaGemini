"""
Script test model trên test set và ảnh đơn lẻ
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from pathlib import Path
import config
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class PneumoniaClassifier(nn.Module):
    """Model DenseNet121"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5, pretrained=False):
        super(PneumoniaClassifier, self).__init__()
        
        from torchvision.models import DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, device='cpu'):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    model = PneumoniaClassifier(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=False
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def get_transform():
    """Get image transform for testing"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(model, image_path, device='cpu'):
    """Predict trên một ảnh đơn"""
    
    transform = get_transform()
    
    # Load và preprocess ảnh
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = ['NORMAL', 'PNEUMONIA']
    predicted_label = class_names[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'probabilities': {
            'NORMAL': probabilities[0][0].item(),
            'PNEUMONIA': probabilities[0][1].item()
        }
    }


def test_on_directory(model, test_dir, device='cpu'):
    """Test model trên toàn bộ thư mục test"""
    
    print("\n" + "=" * 80)
    print("TESTING MODEL ON TEST SET")
    print("=" * 80)
    
    transform = get_transform()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    class_names = ['NORMAL', 'PNEUMONIA']
    
    # Load tất cả ảnh test
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(test_dir) / class_name
        
        if not class_dir.exists():
            print(f"⚠ Directory not found: {class_dir}")
            continue
        
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        print(f"\nTesting {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            try:
                # Load và predict
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                
                all_predictions.append(predicted_class)
                all_labels.append(class_idx)
                all_probabilities.append(probabilities[0].cpu().numpy())
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
    
    # Tính metrics
    print("\n" + "-" * 80)
    print("TEST RESULTS")
    print("-" * 80)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name:.<20} Accuracy: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # Sensitivity và Specificity (cho binary classification)
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f"\nAdditional Metrics:")
        print(f"  Sensitivity (Recall)....... {sensitivity:.4f}")
        print(f"  Specificity................ {specificity:.4f}")
        print(f"  Precision.................. {precision:.4f}")
        print(f"  F1-Score................... {f1:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix_detailed(cm, class_names)
    
    # Plot ROC curve
    if len(class_names) == 2:
        plot_roc_curve(all_labels, np.array(all_probabilities)[:, 1])
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm
    }


def plot_confusion_matrix_detailed(cm, class_names):
    """Vẽ confusion matrix chi tiết với tỷ lệ phần trăm"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute numbers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Plot 2: Percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(config.MODEL_DIR, 'test_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Confusion matrix saved to: {save_path}")


def plot_roc_curve(labels, probabilities):
    """Vẽ ROC curve"""
    
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Pneumonia Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(config.MODEL_DIR, 'test_roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve saved to: {save_path}")
    print(f"  AUC Score: {roc_auc:.4f}")


def interactive_test():
    """Chế độ test tương tác"""
    
    print("\n" + "=" * 80)
    print("INTERACTIVE TESTING MODE")
    print("=" * 80)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(config.MODEL_DIR, 'final_model.pth')
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {config.MODEL_DIR}")
        print("Please train the model first!")
        return
    
    model = load_model(model_path, device)
    
    while True:
        print("\n" + "-" * 80)
        print("Choose an option:")
        print("  1. Test on single image")
        print("  2. Test on entire test set")
        print("  3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("\nEnter image path: ").strip().strip('"')
            
            if not os.path.exists(image_path):
                print(f"✗ Image not found: {image_path}")
                continue
            
            result = predict_single_image(model, image_path, device)
            
            print("\n" + "=" * 60)
            print("PREDICTION RESULT")
            print("=" * 60)
            print(f"Image: {os.path.basename(image_path)}")
            print(f"\nPredicted Class: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print(f"\nProbabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name:.<20} {prob*100:.2f}%")
        
        elif choice == '2':
            test_dir = os.path.join(config.PROCESSED_DATA_DIR, 'test')
            
            if not os.path.exists(test_dir):
                print(f"✗ Test directory not found: {test_dir}")
                continue
            
            results = test_on_directory(model, test_dir, device)
            print(f"\n✓ Testing complete!")
        
        elif choice == '3':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")


def main():
    """Main function"""
    
    print("=" * 80)
    print("PNEUMONIA DETECTION MODEL - TESTING SCRIPT")
    print("=" * 80)
    
    # Check if model exists
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(config.MODEL_DIR, 'final_model.pth')
    
    if not os.path.exists(model_path):
        print(f"\n✗ No trained model found in {config.MODEL_DIR}")
        print("Please train the model first using: python train_model.py")
        return
    
    # Run interactive test
    interactive_test()


if __name__ == "__main__":
    main()
