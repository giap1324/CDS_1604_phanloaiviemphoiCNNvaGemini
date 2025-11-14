"""
Script kiểm tra và debug model predictions
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import config
from train_model import PneumoniaClassifier

def test_model_consistency():
    """Kiểm tra model có hoạt động nhất quán không"""
    print("="*80)
    print("MODEL CONSISTENCY TEST")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"Model training accuracy: {checkpoint.get('train_acc', 0):.4f}")
    print(f"Model validation accuracy: {checkpoint.get('val_acc', 0):.4f}")
    print(f"Epoch: {checkpoint.get('epoch', 0) + 1}\n")
    
    # Tạo model với đúng cấu hình
    model = PneumoniaClassifier(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=True  # QUAN TRỌNG: phải True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # SET EVAL MODE
    model = model.to(device)
    
    print(f"Model is in training mode: {model.training}")
    print(f"Model is in eval mode: {not model.training}\n")
    
    # Test trên một số ảnh từ test set
    test_dir = os.path.join(config.PROCESSED_DATA_DIR, 'test')
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        print(f"\nTesting {class_name} images:")
        print("-"*80)
        
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')][:5]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            # Preprocessing CHÍNH XÁC
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            pred_label = 'NORMAL' if predicted_class == 0 else 'PNEUMONIA'
            is_correct = (pred_label == class_name)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {img_file[:30]:30} | True: {class_name:10} | Pred: {pred_label:10} | Conf: {confidence:.2%}")

def analyze_prediction_distribution():
    """Phân tích phân phối predictions trên test set"""
    print("\n" + "="*80)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = PneumoniaClassifier(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join(config.PROCESSED_DATA_DIR, 'test')
    
    results = {
        'NORMAL': {'correct': 0, 'total': 0, 'confidences': []},
        'PNEUMONIA': {'correct': 0, 'total': 0, 'confidences': []}
    }
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                pred_label = 'NORMAL' if predicted_class == 0 else 'PNEUMONIA'
                
                results[class_name]['total'] += 1
                results[class_name]['confidences'].append(confidence)
                
                if pred_label == class_name:
                    results[class_name]['correct'] += 1
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Print results
    print("\nResults by class:")
    print("-"*80)
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        total = results[class_name]['total']
        correct = results[class_name]['correct']
        accuracy = correct / total * 100 if total > 0 else 0
        avg_conf = np.mean(results[class_name]['confidences']) if results[class_name]['confidences'] else 0
        
        print(f"\n{class_name}:")
        print(f"  Total samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Average confidence: {avg_conf:.2%}")
    
    # Overall
    total_all = sum(r['total'] for r in results.values())
    correct_all = sum(r['correct'] for r in results.values())
    overall_accuracy = correct_all / total_all * 100 if total_all > 0 else 0
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Total samples: {total_all}")

def check_model_weights():
    """Kiểm tra weights của model"""
    print("\n" + "="*80)
    print("MODEL WEIGHTS CHECK")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    state_dict = checkpoint['model_state_dict']
    print(f"\nModel layers: {len(state_dict)}")
    
    # Check một số weights
    for name, param in list(state_dict.items())[:5]:
        print(f"  {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")

if __name__ == "__main__":
    # Test 1: Model consistency
    test_model_consistency()
    
    # Test 2: Prediction distribution
    analyze_prediction_distribution()
    
    # Test 3: Check weights
    check_model_weights()
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nNếu accuracy thấp hơn expected:")
    print("  1. Kiểm tra model có đang ở eval mode không (dropout phải tắt)")
    print("  2. Kiểm tra preprocessing có khớp với training không")
    print("  3. Kiểm tra ảnh input có bị corrupt không")
    print("  4. Xem xét train lại model với data augmentation tốt hơn")
