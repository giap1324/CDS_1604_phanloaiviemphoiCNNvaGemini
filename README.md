<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
        🎓 Faculty of Information Technology - Dai Nam University
    </a>
</h2>

<h2 align="center">
    🩺 PHÂN LOẠI VIÊM PHỔI DỰA TRÊN HÌNH ẢNH X-QUANG <br/>
    (Chest X-Ray Pneumonia Classification using CNN and Gemini API)
</h2>

<div align="center">
    <p align="center">
        <img src="docs/aiotlab_logo.png" alt="AIoTLab Logo" width="170"/>
        <img src="docs/fitdnu_logo.png" alt="FIT Logo" width="180"/>
        <img src="docs/dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>

[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>

---

## 📘 GIỚI THIỆU ĐỀ TÀI

Đề tài **"Phân loại viêm phổi dựa trên hình ảnh X-quang sử dụng mô hình học sâu CNN và API Gemini"** nhằm hỗ trợ **chuẩn đoán bệnh viêm phổi** từ hình ảnh X-quang phổi của bệnh nhân.  
Mục tiêu là **tự động hóa quá trình nhận diện** giữa các lớp bệnh:
- 🧬 **COVID-19 Pneumonia**
- 🦠 **Viral Pneumonia**
- 🧫 **Bacterial Pneumonia**
- 🫁 **Normal**

Mô hình được xây dựng dựa trên kiến trúc **Convolutional Neural Network (CNN)** kết hợp với **Gemini API** để tự động sinh báo cáo chẩn đoán hỗ trợ bác sĩ.

---

## 🧠 MÔ HÌNH HỌC SÂU (CNN)

- Framework: **TensorFlow / Keras**
- Input: Ảnh X-quang (224x224 RGB)
- Augmentation: Rotation, Flip, Brightness, Zoom
- Optimizer: `Adam`, Learning Rate = 0.001  
- Loss Function: `Categorical Crossentropy`
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

<p align="center">
  <img src="docs/Typical-CNN-Architecture-1024x374.png" width="600" alt="CNN Architecture">
</p>

---

## 🧩 KIẾN TRÚC HỆ THỐNG

```text
+-----------------------------+
|       HÌNH ẢNH X-QUANG     |
+-------------+---------------+
              |
              v
   [TIỀN XỬ LÝ DỮ LIỆU - DataPreprocessor]
              |
              v
       [MÔ HÌNH CNN HUẤN LUYỆN]
              |
              v
   [DỰ ĐOÁN KẾT QUẢ - PREDICTION]
              |
              v
   [API GEMINI -> SINH BÁO CÁO TỰ ĐỘNG]
```

---

## 📊 KẾT QUẢ ĐÁNH GIÁ

| Class               | Precision |  Recall  | F1-Score |  Support |
| :------------------ | :-------: | :------: | :------: | :------: |
| COVID-19 Pneumonia  |    0.15   |   0.30   |   0.20   |   1446   |
| Bacterial Pneumonia |    0.07   |   0.11   |   0.08   |   2404   |
| Normal              |    0.22   |   0.05   |   0.08   |   4076   |
| Viral Pneumonia     |    0.97   |   0.71   |   0.82   |    538   |
| **Accuracy**        |           | **0.15** |          | **8464** |

📈 Mặc dù độ chính xác tổng thể chưa cao, hệ thống vẫn nhận diện tốt nhóm **Viral Pneumonia** và có thể cải thiện với dữ liệu cân bằng hơn hoặc huấn luyện thêm trên TPU.

---

## 🔧 CÀI ĐẶT & CHẠY DỰ ÁN

### 1️⃣ Clone project

```bash
git clone https://github.com/username/xray-cnn-gemini.git
cd xray-cnn-gemini
```

### 2️⃣ Cài thư viện

```bash
pip install -r requirements.txt
```

Hoặc nếu chạy trên Google Colab:

```python
!pip install tensorflow==2.13.0 Pillow numpy matplotlib seaborn scikit-learn google-generativeai
```

### 3️⃣ Huấn luyện mô hình

```python
python train_model.py
```

### 4️⃣ Kiểm tra mô hình

```python
python evaluate_model.py
```

---

## 🤖 TÍCH HỢP GEMINI API

Mô hình sau khi huấn luyện có thể gửi kết quả đến **Gemini 2.5 Pro** để:

- Sinh **báo cáo chẩn đoán tự động** (mô tả phát hiện, phân tích vùng ảnh bất thường)
- Gợi ý mức độ nghi ngờ bệnh lý
- Sinh **file PDF** hoặc gửi qua web app Flask

---

## 💡 ĐỊNH HƯỚNG PHÁT TRIỂN

- Nâng cấp mô hình sang **Vision Transformer (ViT)** hoặc **EfficientNet**.
- Tích hợp **Grad-CAM** để trực quan hóa vùng tổn thương.
- Tạo **web dashboard** hiển thị kết quả thời gian thực.
- Tăng tốc huấn luyện bằng **TPU Colab**.

---


## ✉️ 5. Liên hệ

**Tác giả**: Nguyễn Đào Nguyên Giáp 

📧 **Email**: nguyennguyenvh09@gmail.com  
🏫 **Trường**: Đại học Đại Nam - Khoa Công nghệ Thông tin  

---

<p align="center">
  <b>© 2025 Faculty of Information Technology - Dai Nam University</b><br>
  Developed with ❤️ by <b>AIoT Lab</b>
</p>
