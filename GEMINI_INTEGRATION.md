# Hướng dẫn tích hợp Gemini AI

## 📋 Tổng quan

Hệ thống đã được tích hợp API Gemini AI của Google để tự động sinh báo cáo y khoa chi tiết dựa trên kết quả chẩn đoán X-quang phổi.

## 🔧 Cài đặt

### 1. Cài đặt thư viện

Thư viện `google-generativeai` đã được thêm vào `requirements.txt`:

```bash
pip install google-generativeai
```

### 2. Cấu hình API Key

Có 2 cách để cấu hình Gemini API key:

#### Cách 1: Sử dụng biến môi trường (Khuyên dùng)

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY = "your-actual-api-key-here"
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-actual-api-key-here"
```

#### Cách 2: Chỉnh sửa trực tiếp file config.py

Mở file `config.py` và thay đổi:
```python
GEMINI_API_KEY = 'your-actual-api-key-here'
```

### 3. Lấy Gemini API Key

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập bằng tài khoản Google
3. Nhấn "Create API Key"
4. Copy API key và cấu hình theo hướng dẫn trên

## 📁 Cấu trúc mới

### File mới được tạo:

- **gemini_service.py**: Service xử lý gọi API Gemini và sinh báo cáo

### File đã được cập nhật:

- **config.py**: Thêm cấu hình GEMINI_API_KEY
- **models.py**: Thêm trường `medical_report` và `report_generated_at`
- **app.py**: Tích hợp Gemini service vào quy trình chẩn đoán
- **templates/diagnosis.html**: Hiển thị báo cáo y khoa
- **static/style.css**: CSS cho báo cáo y khoa

## 🚀 Cách sử dụng

### 1. Cập nhật Database Schema

Khi chạy lần đầu sau khi cập nhật, cần migrate database:

```python
from app import app, db
from models import Diagnosis

with app.app_context():
    # Tạo lại database với schema mới
    db.drop_all()  # Cẩn thận: Xóa dữ liệu cũ
    db.create_all()
    print("Database updated successfully!")
```

Hoặc chạy script đơn giản:

```bash
python -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database updated!')"
```

### 2. Chạy ứng dụng

```bash
python app.py
```

### 3. Test chức năng

1. Upload ảnh X-quang
2. Xem kết quả chẩn đoán
3. Báo cáo y khoa sẽ tự động được hiển thị ở trang chi tiết

## 🔍 Tính năng Gemini AI

### Báo cáo y khoa bao gồm:

1. **Kết quả chẩn đoán**: Tóm tắt kết quả
2. **Mô tả chi tiết**: Đặc điểm quan sát từ X-quang
3. **Đánh giá**: Phân tích tình trạng
4. **Khuyến nghị**: Hướng dẫn cho bệnh nhân
5. **Lưu ý quan trọng**: Cảnh báo về giới hạn của AI

### Chế độ fallback

Nếu không có API key hoặc API lỗi, hệ thống sẽ tự động tạo báo cáo mặc định với nội dung cơ bản.

## 📊 Flow hoạt động

```
Upload ảnh → AI phân tích → Dự đoán (NORMAL/PNEUMONIA)
                ↓
          Gemini API được gọi
                ↓
      Sinh báo cáo y khoa chi tiết
                ↓
         Lưu vào database
                ↓
      Hiển thị kết quả + báo cáo
```

## ⚙️ Tùy chỉnh

### Thay đổi prompt của Gemini

Chỉnh sửa method `_create_medical_prompt()` trong `gemini_service.py`:

```python
def _create_medical_prompt(self, prediction, confidence, patient_info):
    prompt = f"""
    [Tùy chỉnh prompt của bạn ở đây]
    
    Kết quả: {prediction}
    Độ tin cậy: {confidence}%
    """
    return prompt
```

### Thêm thông tin bệnh nhân

Cập nhật `app.py` để truyền thêm thông tin:

```python
medical_report = gemini_service.generate_medical_report(
    prediction=predicted_class,
    confidence=confidence,
    patient_info={
        'name': 'Nguyễn Văn A',
        'age': 35,
        'gender': 'Nam',
        'symptoms': 'Ho, sốt cao'
    }
)
```

## 🛡️ Bảo mật

- **KHÔNG** commit API key lên Git
- Sử dụng biến môi trường cho production
- Giới hạn rate limit nếu cần
- Validate input trước khi gửi lên Gemini

## 🐛 Troubleshooting

### Lỗi: "API key not configured"

**Giải pháp**: Đặt biến môi trường GEMINI_API_KEY

### Lỗi: "quota exceeded"

**Giải pháp**: Kiểm tra quota tại Google AI Studio hoặc nâng cấp plan

### Lỗi: Database migration

**Giải pháp**: Chạy lại `db.create_all()` hoặc sử dụng Flask-Migrate

### Báo cáo không hiển thị

**Giải pháp**: 
1. Kiểm tra database có trường `medical_report`
2. Kiểm tra API key có hợp lệ
3. Xem logs để tìm lỗi

## 📝 Ghi chú

- Gemini API có giới hạn requests miễn phí
- Thời gian tạo báo cáo: 2-5 giây
- Báo cáo được lưu vào database để tái sử dụng
- Hỗ trợ cả tiếng Việt và tiếng Anh

## 🔗 Tài liệu tham khảo

- [Google AI Studio](https://makersuite.google.com)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Python SDK](https://github.com/google/generative-ai-python)

## ✅ Checklist triển khai

- [x] Cài đặt google-generativeai
- [x] Tạo gemini_service.py
- [x] Cập nhật config.py
- [x] Cập nhật models.py
- [x] Tích hợp vào app.py
- [x] Cập nhật templates
- [x] Cập nhật CSS
- [ ] Lấy Gemini API key
- [ ] Cấu hình API key
- [ ] Migrate database
- [ ] Test chức năng

---

**Lưu ý**: Đây là tích hợp AI hỗ trợ, KHÔNG thay thế chẩn đoán y khoa chính thức của bác sĩ!
