import os
from flask import Flask, request, render_template, jsonify, url_for
import cv2
import numpy as np
import pytesseract
from PIL import Image
import base64
import easyocr

app = Flask(__name__)

# Cấu hình đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

reader = easyocr.Reader(['vi', 'en'])

def preprocess_image(image):
    # Nếu ảnh là grayscale (2 chiều), không chuyển đổi gì cả
    if len(image.shape) == 2:
        gray = image
    # Nếu ảnh có alpha channel (4 kênh), chuyển sang GRAY
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    # Nếu ảnh là RGB (3 kênh), chuyển sang GRAY
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format")

    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Giảm nhiễu
    denoised = cv2.fastNlMeansDenoising(gray)

    # Áp dụng Gaussian blur để làm mịn
    blur = cv2.GaussianBlur(denoised, (3, 3), 0)

    # Áp dụng adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

    # Tăng kích thước ảnh (nếu cần)
    scale_percent = 200
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

    return resized

def preprocess_image_for_easyocr(image):
    # Nếu ảnh là RGBA, chuyển sang RGB
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # Nếu ảnh là grayscale, chuyển sang RGB
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Nếu ảnh là RGB, giữ nguyên
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Đọc ảnh
            img = Image.open(file.stream)
            img_np = np.array(img)
            
            # Tiền xử lý ảnh cho EasyOCR
            processed_img = preprocess_image_for_easyocr(img_np)
            
            # Thực hiện OCR với EasyOCR
            result = reader.readtext(processed_img, detail=0, paragraph=True)
            text = '\n'.join(result)
            
            # Tạo tên file duy nhất
            filename = f"processed_{file.filename}"
            processed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Lưu ảnh đã xử lý (ảnh gốc hoặc RGB)
            cv2.imwrite(processed_img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            
            # Tạo URL cho ảnh đã xử lý
            processed_image_url = url_for('static', filename=f'uploads/{filename}')
            
            return jsonify({
                'text': text,
                'processed_image': processed_image_url
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 