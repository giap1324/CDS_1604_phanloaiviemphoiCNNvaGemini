"""
Configuration file for Pneumonia Classification Project
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
TEMPLATES_FOLDER = os.path.join(BASE_DIR, 'templates')

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, 
                 UPLOAD_FOLDER, STATIC_FOLDER, TEMPLATES_FOLDER]:
    os.makedirs(dir_path, exist_ok=True)

# Image processing
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3  # RGB

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Model parameters
NUM_CLASSES = 2  # NORMAL, PNEUMONIA
BATCH_SIZE = 16  # Giảm xuống 16 cho GTX 1650 4GB
EPOCHS = 15  # Giảm từ 20 xuống 15 (thường đủ với early stopping)
LEARNING_RATE = 0.0005  # Giảm một chút để ổn định hơn
DROPOUT_RATE = 0.5

# Database
DATABASE_URI = 'sqlite:///pneumonia_diagnosis.db'

# Flask
SECRET_KEY = 'your-secret-key-here-change-in-production'
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

# Gemini AI Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'your-gemini-api-key-here')
# Để sử dụng: Đặt biến môi trường GEMINI_API_KEY hoặc thay 'your-gemini-api-key-here' bằng API key thực

