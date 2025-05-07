# OCR Image to Text Converter

Đây là một ứng dụng web đơn giản để chuyển đổi hình ảnh thành văn bản sử dụng OCR (Optical Character Recognition).

## Tính năng

- Chuyển đổi hình ảnh thành văn bản
- Hỗ trợ tiếng Việt và tiếng Anh
- Tiền xử lý ảnh để cải thiện độ chính xác
- Giao diện web thân thiện với người dùng

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Tesseract OCR
- Các thư viện Python (xem requirements.txt)

## Cài đặt

1. Cài đặt Tesseract OCR:
   - Windows: Tải và cài đặt từ https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

2. Cài đặt các thư viện Python:
   ```bash
   pip install -r requirements.txt
   ```

3. Tải thêm ngôn ngữ cho Tesseract (nếu cần):
   - Windows: Tải file ngôn ngữ từ https://github.com/tesseract-ocr/tessdata
   - Linux/Mac: `sudo apt-get install tesseract-ocr-vie`

## Chạy ứng dụng

1. Chạy server:
   ```bash
   python app.py
   ```

2. Mở trình duyệt và truy cập: http://localhost:5000

## Sử dụng

1. Nhấn nút "Chọn ảnh" để chọn file ảnh cần chuyển đổi
2. Nhấn "Chuyển đổi" để bắt đầu quá trình OCR
3. Kết quả sẽ hiển thị bên dưới, bao gồm:
   - Ảnh gốc
   - Ảnh đã xử lý
   - Văn bản được trích xuất

## Lưu ý

- Ứng dụng hoạt động tốt nhất với ảnh có độ tương phản cao
- Kết quả có thể khác nhau tùy thuộc vào chất lượng ảnh đầu vào
- Hỗ trợ các định dạng ảnh phổ biến: JPG, PNG, BMP 