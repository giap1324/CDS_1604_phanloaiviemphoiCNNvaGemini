"""
Flask Web Application for Pneumonia Classification
- Upload ảnh X-quang → hiển thị dự đoán
- Database lưu kết quả và lịch sử chẩn đoán
"""
import os
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from datetime import datetime
import config
from models import db, Diagnosis
from gemini_service import get_gemini_service

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE

# Initialize database
db.init_app(app)

# Custom Jinja2 filter for markdown-like formatting
@app.template_filter('format_medical_report')
def format_medical_report(text):
    """Convert markdown-like text to HTML"""
    if not text:
        return ''
    
    # Replace ## headers
    text = re.sub(r'^## (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    
    # Replace **bold** text
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Replace line breaks
    text = text.replace('\n\n', '</p><p>')
    text = text.replace('\n', '<br>')
    
    # Wrap in paragraph if not already wrapped
    if not text.startswith('<h4>') and not text.startswith('<p>'):
        text = '<p>' + text
    if not text.endswith('</p>') and not text.endswith('</h4>'):
        text = text + '</p>'
    
    return text

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

class PneumoniaClassifier(nn.Module):
    """Model DenseNet121 với dropout"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5, pretrained=True):
        super(PneumoniaClassifier, self).__init__()
        self.backbone = models.densenet121(pretrained=pretrained)
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

def load_model():
    """Load trained model"""
    global model
    model_path = os.path.join(config.MODEL_DIR, 'final_model.pth')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = PneumoniaClassifier(
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            pretrained=True  # Sửa thành True để khớp với training
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Quan trọng: set eval mode trước khi to(device)
        model = model.to(device)
        print(f"Model loaded from {model_path}")
        print(f"Model in eval mode: {not model.training}")
        return True
    else:
        print(f"Model not found at {model_path}")
        return False

def preprocess_image(image_path):
    """Tiền xử lý ảnh để đưa vào model - PHẢI KHỚP VỚI TRAINING"""
    # Transform CHÍNH XÁC như trong training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize cố định 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load và convert sang RGB
    image = Image.open(image_path).convert('RGB')
    
    # Transform và add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor.to(device)

def predict_image(image_path):
    """Dự đoán ảnh"""
    if model is None:
        return None, None
    
    try:
        image_tensor = preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            class_names = ['NORMAL', 'PNEUMONIA']
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

def allowed_file(filename):
    """Kiểm tra file extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Trang chủ"""
    # Lấy 10 chẩn đoán gần nhất
    recent_diagnoses = Diagnosis.query.order_by(
        Diagnosis.timestamp.desc()
    ).limit(10).all()
    
    return render_template('index.html', recent_diagnoses=recent_diagnoses, config=config)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Xử lý upload ảnh và dự đoán"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        # Lưu file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Dự đoán
        predicted_class, confidence = predict_image(filepath)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Sinh báo cáo y khoa bằng Gemini AI
        gemini_service = get_gemini_service()
        medical_report = gemini_service.generate_medical_report(
            prediction=predicted_class,
            confidence=confidence
        )
        
        # Lưu vào database
        diagnosis = Diagnosis(
            filename=filename,
            filepath=filepath,
            prediction=predicted_class,
            confidence=confidence,
            timestamp=datetime.now(),
            medical_report=medical_report,
            report_generated_at=datetime.now() if medical_report else None
        )
        db.session.add(diagnosis)
        db.session.commit()
        
        # Trả về kết quả
        result = {
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'is_normal': predicted_class == 'NORMAL',
            'diagnosis_id': diagnosis.id,
            'has_report': medical_report is not None
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """Trang lịch sử chẩn đoán"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    diagnoses = Diagnosis.query.order_by(
        Diagnosis.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('history.html', diagnoses=diagnoses)

@app.route('/diagnosis/<int:diagnosis_id>')
def view_diagnosis(diagnosis_id):
    """Xem chi tiết một chẩn đoán"""
    diagnosis = Diagnosis.query.get_or_404(diagnosis_id)
    return render_template('diagnosis.html', diagnosis=diagnosis, config=config)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Phục vụ ảnh đã upload"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/stats')
def get_stats():
    """API lấy thống kê"""
    total = Diagnosis.query.count()
    normal_count = Diagnosis.query.filter_by(prediction='NORMAL').count()
    pneumonia_count = Diagnosis.query.filter_by(prediction='PNEUMONIA').count()
    
    return jsonify({
        'total': total,
        'normal': normal_count,
        'pneumonia': pneumonia_count,
        'normal_percentage': round(normal_count / total * 100, 2) if total > 0 else 0,
        'pneumonia_percentage': round(pneumonia_count / total * 100, 2) if total > 0 else 0
    })

if __name__ == '__main__':
    # Tạo database tables
    with app.app_context():
        db.create_all()
    
    # Load model
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("ERROR: Model not found. Please train the model first using train_model.py")

