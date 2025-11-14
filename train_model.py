"""
Training Script for Pneumonia Classification
- Baseline: DenseNet121
- Fine-tuning với dropout để chống overfitting
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.amp import autocast, GradScaler
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config

class PneumoniaDataset(Dataset):
    """Dataset class cho ảnh X-quang phổi"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load ảnh và nhãn
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(0 if class_name == 'NORMAL' else 1)
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
        print(f"  - NORMAL: {self.labels.count(0)}")
        print(f"  - PNEUMONIA: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load ảnh với LANCZOS resampling nhanh hơn
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """Tạo data transforms với augmentation cho training"""
    
    # Data augmentation cho training - giảm bớt để nhanh hơn
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize trước để tăng tốc
        transforms.RandomHorizontalFlip(p=0.3),  # Giảm xác suất
        transforms.RandomRotation(degrees=10),  # Giảm độ xoay
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Transform cho validation và test (không augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class PneumoniaClassifier(nn.Module):
    """Model DenseNet121 với dropout để chống overfitting"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5, pretrained=True):
        super(PneumoniaClassifier, self).__init__()
        
        # Load DenseNet121 pretrained
        from torchvision.models import DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        # Thay đổi classifier cuối cùng
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

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Training một epoch với mixed precision"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    use_amp = scaler is not None and device.type == 'cuda'
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Nhanh hơn zero_grad()
        
        # Mixed precision training
        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validation một epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Vẽ biểu đồ lịch sử training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    """Hàm chính để training model"""
    
    print("=" * 60)
    print("TRAINING PNEUMONIA CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Đường dẫn dữ liệu
    data_dir = config.PROCESSED_DATA_DIR
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Kiểm tra dữ liệu
    if not os.path.exists(train_dir):
        print(f"Không tìm thấy dữ liệu training tại {train_dir}")
        print("Vui lòng chạy data_preprocessing.py trước!")
        return
    
    # Tạo datasets
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = PneumoniaDataset(train_dir, transform=train_transform)
    val_dataset = PneumoniaDataset(val_dir, transform=val_transform)
    test_dataset = PneumoniaDataset(test_dir, transform=val_transform)
    
    # Tạo dataloaders với tối ưu hóa cho Windows
    num_workers = 2  # Giảm num_workers trên Windows để tránh bottleneck
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # Tăng tốc transfer sang GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE * 2,  # Batch lớn hơn cho validation
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE * 2, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Tạo model
    model = PneumoniaClassifier(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=True
    )
    model = model.to(device)
    
    # torch.compile() không hoạt động tốt trên Windows, tắt để tránh lỗi Triton
    # model = torch.compile(model)  # Uncomment nếu dùng Linux
    print("Using eager mode (torch.compile disabled on Windows)")
    
    # Loss và optimizer với AdamW (tốt hơn Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed precision scaler cho GPU
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training (FP16)")
    
    # Training
    print("\nBắt đầu training...")
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 60)
        
        # Train với mixed precision
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, model_path)
            print(f"Model saved! (Val Acc: {val_acc:.4f})")
    
    # Vẽ biểu đồ training history
    history_path = os.path.join(config.MODEL_DIR, 'training_history.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, history_path)
    
    # Test trên test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    # Load model tốt nhất
    checkpoint = torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Classification report
    class_names = ['NORMAL', 'PNEUMONIA']
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion matrix
    cm_path = os.path.join(config.MODEL_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    # Lưu model cuối cùng
    final_model_path = os.path.join(config.MODEL_DIR, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'num_classes': config.NUM_CLASSES,
    }, final_model_path)
    
    print(f"\nModel saved to {final_model_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()

