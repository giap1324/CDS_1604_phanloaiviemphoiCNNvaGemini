"""
Database Models
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Diagnosis(db.Model):
    """Model lưu lịch sử chẩn đoán"""
    __tablename__ = 'diagnoses'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # NORMAL hoặc PNEUMONIA
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now, nullable=False)
    
    # Thêm trường báo cáo y khoa từ Gemini AI
    medical_report = db.Column(db.Text, nullable=True)
    report_generated_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<Diagnosis {self.id}: {self.prediction} ({self.confidence:.2f}%)>'
    
    def to_dict(self):
        """Chuyển đổi sang dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'prediction': self.prediction,
            'confidence': round(self.confidence, 2),
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'medical_report': self.medical_report,
            'report_generated_at': self.report_generated_at.strftime('%Y-%m-%d %H:%M:%S') if self.report_generated_at else None
        }

