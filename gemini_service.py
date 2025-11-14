"""
Gemini AI Service for generating medical reports
"""
import google.generativeai as genai
import os
from typing import Optional, Dict
import config

class GeminiReportGenerator:
    """Dịch vụ sinh báo cáo y khoa bằng Gemini AI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo Gemini service
        
        Args:
            api_key: API key của Google Gemini (nếu None sẽ lấy từ config)
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        if self.api_key and self.api_key != "your-gemini-api-key-here":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            print("WARNING: Gemini API key not configured. Reports will not be generated.")
    
    def is_configured(self) -> bool:
        """Kiểm tra xem Gemini đã được cấu hình chưa"""
        return self.model is not None
    
    def generate_medical_report(
        self, 
        prediction: str, 
        confidence: float,
        patient_info: Optional[Dict] = None
    ) -> str:
        """
        Sinh báo cáo y khoa dựa trên kết quả dự đoán
        
        Args:
            prediction: Kết quả dự đoán (NORMAL hoặc PNEUMONIA)
            confidence: Độ tin cậy của dự đoán (%)
            patient_info: Thông tin bệnh nhân (tùy chọn)
        
        Returns:
            Báo cáo y khoa dưới dạng text
        """
        if not self.is_configured():
            return self._generate_default_report(prediction, confidence)
        
        try:
            # Chuẩn bị prompt cho Gemini
            prompt = self._create_medical_prompt(prediction, confidence, patient_info)
            
            # Gọi Gemini API
            response = self.model.generate_content(prompt)
            
            # Lấy text từ response
            if response and response.text:
                return response.text
            else:
                return self._generate_default_report(prediction, confidence)
                
        except Exception as e:
            print(f"Error generating Gemini report: {e}")
            return self._generate_default_report(prediction, confidence)
    
    def _create_medical_prompt(
        self, 
        prediction: str, 
        confidence: float,
        patient_info: Optional[Dict] = None
    ) -> str:
        """Tạo prompt cho Gemini AI"""
        
        # Thông tin cơ bản
        patient_section = ""
        if patient_info:
            patient_section = f"""
THÔNG TIN BỆNH NHÂN (nếu có):
{self._format_patient_info(patient_info)}
"""
        
        prompt = f"""Bạn là một bác sĩ chuyên khoa hình ảnh y khoa có kinh nghiệm. Hãy viết một báo cáo y khoa chuyên nghiệp dựa trên kết quả phân tích X-quang phổi sau:

KẾT QUẢ PHÂN TÍCH TỰ ĐỘNG:
- Chẩn đoán: {prediction}
- Độ tin cậy: {confidence:.2f}%

{patient_section}

Hãy viết báo cáo y khoa theo cấu trúc sau:

## KẾT QUẢ CHẨN ĐOÁN

[Nêu rõ kết quả chẩn đoán]

## MÔ TẢ CHI TIẾT

[Mô tả các đặc điểm quan sát được từ ảnh X-quang]

## ĐÁNH GIÁ

[Đánh giá tình trạng dựa trên kết quả]

## KHUYẾN NGHỊ

[Đưa ra các khuyến nghị phù hợp với kết quả]

## LƯU Ý QUAN TRỌNG

[Các lưu ý cần thiết cho bệnh nhân và bác sĩ điều trị]

Lưu ý:
- Sử dụng thuật ngữ y khoa chuyên nghiệp nhưng dễ hiểu
- Giải thích rõ ràng các phát hiện
- Đưa ra khuyến nghị cụ thể và hữu ích
- Nhấn mạnh rằng đây là kết quả từ AI cần được xác nhận bởi bác sĩ
- Độ tin cậy {confidence:.2f}% {"cao" if confidence >= 90 else "trung bình" if confidence >= 70 else "thấp"}, cần xem xét kỹ lưỡng
"""
        
        return prompt
    
    def _format_patient_info(self, patient_info: Dict) -> str:
        """Format thông tin bệnh nhân"""
        info_lines = []
        if patient_info.get('name'):
            info_lines.append(f"- Họ tên: {patient_info['name']}")
        if patient_info.get('age'):
            info_lines.append(f"- Tuổi: {patient_info['age']}")
        if patient_info.get('gender'):
            info_lines.append(f"- Giới tính: {patient_info['gender']}")
        if patient_info.get('symptoms'):
            info_lines.append(f"- Triệu chứng: {patient_info['symptoms']}")
        
        return "\n".join(info_lines) if info_lines else "Không có thông tin"
    
    def _generate_default_report(self, prediction: str, confidence: float) -> str:
        """Sinh báo cáo mặc định khi không có Gemini API"""
        
        if prediction == "PNEUMONIA":
            return f"""## KẾT QUẢ CHẨN ĐOÁN

Phân tích hình ảnh X-quang cho thấy các dấu hiệu gợi ý **VIÊM PHỔI (PNEUMONIA)** với độ tin cậy {confidence:.2f}%.

## MÔ TẢ CHI TIẾT

Hệ thống AI phát hiện các đặc điểm bất thường trên ảnh X-quang phổi có thể liên quan đến tình trạng viêm nhiễm, bao gồm:
- Tăng đậm mật độ tổ chức phổi
- Các vùng đục hoặc thâm nhiễm có thể quan sát được
- Các dấu hiệu tương thích với viêm phổi

## ĐÁNH GIÁ

Kết quả phân tích tự động cho thấy khả năng cao có viêm phổi. Tuy nhiên, chẩn đoán cuối cùng cần được xác nhận bởi bác sĩ chuyên khoa dựa trên:
- Đánh giá lâm sàng trực tiếp
- Các triệu chứng và dấu hiệu sinh học
- Xét nghiệm bổ sung nếu cần thiết

## KHUYẾN NGHỊ

1. **Gấp rút**: Liên hệ với bác sĩ chuyên khoa càng sớm càng tốt
2. **Không tự ý điều trị**: Chờ chẩn đoán chính thức từ bác sĩ
3. **Theo dõi triệu chứng**: Khó thở, sốt cao, đau ngực - cần đi khám ngay
4. **Chuẩn bị thông tin**: Ghi lại triệu chứng, thời gian xuất hiện để báo bác sĩ

## LƯU Ý QUAN TRỌNG

⚠️ **Đây chỉ là kết quả sàng lọc ban đầu bằng AI, KHÔNG thay thế chẩn đoán y khoa chính thức.**

- Kết quả cần được xác nhận bởi bác sĩ chuyên khoa X-quang
- Độ tin cậy {confidence:.2f}% chỉ mang tính tham khảo
- Hãy đến cơ sở y tế để được khám và điều trị phù hợp
- Trong trường hợp khẩn cấp, gọi cấp cứu ngay lập tức
"""
        else:  # NORMAL
            return f"""## KẾT QUẢ CHẨN ĐOÁN

Phân tích hình ảnh X-quang cho thấy **KHÔNG phát hiện dấu hiệu bất thường rõ ràng** với độ tin cậy {confidence:.2f}%.

## MÔ TẢ CHI TIẾT

Hệ thống AI đánh giá ảnh X-quang phổi và nhận thấy:
- Các trường phổi trong sáng bình thường
- Không có vùng đục hoặc thâm nhiễm rõ ràng
- Cấu trúc phổi có vẻ bình thường
- Không có dấu hiệu viêm nhiễm rõ ràng

## ĐÁNH GIÁ

Kết quả phân tích tự động cho thấy ảnh X-quang phổi trong giới hạn bình thường. Tuy nhiên:
- Kết quả này cần được xác nhận bởi bác sĩ chuyên khoa
- Một số bệnh lý có thể không biểu hiện rõ trên X-quang
- Chẩn đoán y khoa cần kết hợp nhiều yếu tố lâm sàng

## KHUYẾN NGHỊ

1. **Theo dõi sức khỏe**: Tiếp tục theo dõi các triệu chứng nếu có
2. **Khám định kỳ**: Nếu có triệu chứng ho, khó thở, vẫn nên đi khám bác sĩ
3. **Tái khám nếu cần**: Nếu triệu chứng không thuyên giảm hoặc nặng hơn
4. **Xét nghiệm bổ sung**: Bác sĩ có thể chỉ định thêm xét nghiệm nếu cần

## LƯU Ý QUAN TRỌNG

⚠️ **Đây chỉ là kết quả sàng lọc ban đầu bằng AI, KHÔNG thay thế chẩn đoán y khoa chính thức.**

- Kết quả "bình thường" từ AI không đảm bảo 100% không có bệnh
- Nếu có triệu chứng lâm sàng, vẫn cần đi khám bác sĩ
- Độ tin cậy {confidence:.2f}% chỉ mang tính tham khảo
- Chỉ bác sĩ mới có thể đưa ra chẩn đoán chính thức
"""

# Singleton instance
_gemini_service = None

def get_gemini_service() -> GeminiReportGenerator:
    """Lấy singleton instance của GeminiReportGenerator"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiReportGenerator()
    return _gemini_service
