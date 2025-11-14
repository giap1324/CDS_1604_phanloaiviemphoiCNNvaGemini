"""
Setup script để khởi tạo dự án
"""
import os
import config

def create_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODEL_DIR,
        config.UPLOAD_FOLDER,
        config.STATIC_FOLDER,
        config.TEMPLATES_FOLDER,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Đã tạo thư mục: {directory}")

if __name__ == "__main__":
    print("=" * 60)
    print("KHỞI TẠO DỰ ÁN PHÂN LOẠI VIÊM PHỔI")
    print("=" * 60)
    print("\nĐang tạo cấu trúc thư mục...")
    create_directories()
    print("\n✓ Hoàn thành khởi tạo!")
    print("\nBước tiếp theo:")
    print("1. Tải dữ liệu từ Kaggle và đặt vào data/raw/")
    print("2. Chạy: python data_preprocessing.py")
    print("3. Chạy: python train_model.py")
    print("4. Chạy: python app.py")

