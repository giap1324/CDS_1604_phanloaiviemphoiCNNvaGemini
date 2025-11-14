"""
Data Preprocessing Script
- Kiểm tra và xóa ảnh hỏng, kích thước bất thường
- Chuẩn hóa kích thước (224×224), chuyển ảnh sang RGB
- Tăng cường dữ liệu (Data Augmentation)
- Chia dữ liệu train/test/validation (70/20/10)
"""
import os
import sys
import shutil
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_image_validity(image_path):
    """
    Kiểm tra ảnh có hợp lệ không
    Returns: (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it's a valid image
        return True, None
    except Exception as e:
        return False, str(e)

def check_image_size(image_path, min_size=(50, 50), max_size=(5000, 5000)):
    """
    Kiểm tra kích thước ảnh có bất thường không
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width < min_size[0] or height < min_size[1]:
                return False, f"Image too small: {width}x{height}"
            if width > max_size[0] or height > max_size[1]:
                return False, f"Image too large: {width}x{height}"
            return True, None
    except Exception as e:
        return False, str(e)

def clean_images(source_dir, target_dir):
    """
    Làm sạch ảnh: kiểm tra và xóa ảnh hỏng, kích thước bất thường
    """
    print("Bắt đầu làm sạch dữ liệu...")
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    corrupted_count = 0
    abnormal_size_count = 0
    valid_count = 0
    
    # Duyệt qua tất cả các ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(source_path.rglob(f'*{ext}'))
        all_images.extend(source_path.rglob(f'*{ext.upper()}'))
    
    print(f"Tìm thấy {len(all_images)} ảnh để kiểm tra...")
    
    for img_path in tqdm(all_images, desc="Kiểm tra ảnh"):
        # Kiểm tra tính hợp lệ
        is_valid, error = check_image_validity(img_path)
        if not is_valid:
            corrupted_count += 1
            print(f"Ảnh hỏng: {img_path} - {error}")
            continue
        
        # Kiểm tra kích thước
        size_valid, size_error = check_image_size(img_path)
        if not size_valid:
            abnormal_size_count += 1
            print(f"Kích thước bất thường: {img_path} - {size_error}")
            continue
        
        # Copy ảnh hợp lệ vào thư mục đích, giữ nguyên cấu trúc thư mục
        relative_path = img_path.relative_to(source_path)
        target_file = target_path / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(img_path, target_file)
        valid_count += 1
    
    print(f"\nKết quả làm sạch:")
    print(f"  - Ảnh hợp lệ: {valid_count}")
    print(f"  - Ảnh hỏng: {corrupted_count}")
    print(f"  - Kích thước bất thường: {abnormal_size_count}")
    
    return valid_count, corrupted_count, abnormal_size_count

def normalize_images(source_dir, target_dir):
    """
    Chuẩn hóa ảnh: resize về 224x224, chuyển sang RGB
    """
    print("\nBắt đầu chuẩn hóa ảnh...")
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(source_path.rglob(f'*{ext}'))
        all_images.extend(source_path.rglob(f'*{ext.upper()}'))
    
    print(f"Chuẩn hóa {len(all_images)} ảnh...")
    
    for img_path in tqdm(all_images, desc="Chuẩn hóa"):
        try:
            with Image.open(img_path) as img:
                # Chuyển sang RGB nếu cần
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize về 224x224
                img_resized = img.resize(config.IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # Lưu ảnh đã chuẩn hóa
                relative_path = img_path.relative_to(source_path)
                target_file = target_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Lưu dưới dạng JPEG để tiết kiệm dung lượng
                if target_file.suffix.lower() not in ['.jpg', '.jpeg']:
                    target_file = target_file.with_suffix('.jpg')
                
                img_resized.save(target_file, 'JPEG', quality=95)
        except Exception as e:
            print(f"Lỗi khi chuẩn hóa {img_path}: {e}")
    
    print("Hoàn thành chuẩn hóa ảnh!")

def split_data(source_dir, output_dir):
    """
    Chia dữ liệu thành train/val/test (70/20/10)
    """
    print("\nBắt đầu chia dữ liệu...")
    
    from sklearn.model_selection import train_test_split
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Tạo các thư mục
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Thu thập tất cả ảnh và nhãn
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = []
    all_labels = []
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = source_path / class_name
        if not class_dir.exists():
            # Thử tìm trong các thư mục con
            for subdir in source_path.iterdir():
                if subdir.is_dir() and class_name.lower() in subdir.name.lower():
                    class_dir = subdir
                    break
        
        if class_dir.exists():
            for ext in image_extensions:
                images = list(class_dir.rglob(f'*{ext}'))
                images.extend(list(class_dir.rglob(f'*{ext.upper()}')))
                for img_path in images:
                    all_images.append(img_path)
                    all_labels.append(class_name)
    
    if len(all_images) == 0:
        print("Không tìm thấy ảnh! Vui lòng kiểm tra cấu trúc thư mục.")
        print("Cấu trúc mong đợi: data/raw/NORMAL/ và data/raw/PNEUMONIA/")
        return
    
    print(f"Tìm thấy {len(all_images)} ảnh")
    print(f"  - NORMAL: {all_labels.count('NORMAL')}")
    print(f"  - PNEUMONIA: {all_labels.count('PNEUMONIA')}")
    
    # Chia train/test trước (70/30)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, 
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=all_labels,
        random_state=42
    )
    
    # Chia val/test từ phần còn lại (20/10 từ tổng = 66.7/33.3 từ temp)
    val_size = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        stratify=y_temp,
        random_state=42
    )
    
    # Copy ảnh vào các thư mục tương ứng
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    for split_name, (images, labels) in splits.items():
        print(f"\nCopying {split_name} set ({len(images)} images)...")
        for img_path, label in tqdm(zip(images, labels), total=len(images), desc=f"  {split_name}"):
            target_file = output_path / split_name / label / img_path.name
            shutil.copy2(img_path, target_file)
    
    print("\nHoàn thành chia dữ liệu!")
    print(f"  - Train: {len(X_train)} ảnh")
    print(f"  - Val: {len(X_val)} ảnh")
    print(f"  - Test: {len(X_test)} ảnh")

if __name__ == "__main__":
    # Đường dẫn thư mục dữ liệu
    # Giả sử dữ liệu được tải về vào data/raw/
    raw_data_dir = config.RAW_DATA_DIR
    cleaned_data_dir = os.path.join(config.DATA_DIR, 'cleaned')
    normalized_data_dir = os.path.join(config.DATA_DIR, 'normalized')
    final_data_dir = config.PROCESSED_DATA_DIR
    
    print("=" * 60)
    print("CHƯƠNG TRÌNH XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    
    # Bước 1: Làm sạch ảnh
    if os.path.exists(raw_data_dir) and any(os.listdir(raw_data_dir)):
        clean_images(raw_data_dir, cleaned_data_dir)
        
        # Bước 2: Chuẩn hóa ảnh
        normalize_images(cleaned_data_dir, normalized_data_dir)
        
        # Bước 3: Chia dữ liệu
        split_data(normalized_data_dir, final_data_dir)
    else:
        print(f"\nKhông tìm thấy dữ liệu trong {raw_data_dir}")
        print("Vui lòng tải dữ liệu từ Kaggle và đặt vào thư mục data/raw/")
        print("\nCấu trúc thư mục mong đợi:")
        print("  data/raw/")
        print("    ├── NORMAL/")
        print("    └── PNEUMONIA/")

