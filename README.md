# Phân loại nguy cơ tái nhập viện của bệnh nhân tiểu đường
Mục tiêu
Phát hiện sớm bệnh nhân có nguy cơ tái nhập viện
Hỗ trợ bác sĩ đưa ra quyết định điều trị
Giảm chi phí và tải cho hệ thống y tế
## 📁 Cấu trúc thư mục

```
taikham_diabetes/
│
├── data/
│   └── diabetic_data.csv          # Dữ liệu gốc
│
├── models/
│   └── lightgbm_best.pkl          # Model đã train
│
├── app.py                         # Ứng dụng Streamlit
├── requirements.txt               # Thư viện cần cài
├── README.md                      # Tài liệu mô tả
```