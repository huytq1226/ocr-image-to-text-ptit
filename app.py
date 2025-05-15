import os
from flask import Flask, request, render_template, jsonify, url_for
import cv2
import numpy as np
import pytesseract
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import base64
import jiwer
import pandas as pd
import easyocr

app = Flask(__name__)

# Cấu hình đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Cấu hình thư mục
UPLOAD_FOLDER = 'static/uploads'
INTERMEDIATE_FOLDER = 'static/intermediate'
for folder in [UPLOAD_FOLDER, INTERMEDIATE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INTERMEDIATE_FOLDER'] = INTERMEDIATE_FOLDER
# Khởi tạo EasyOCR
reader = easyocr.Reader(['vi', 'en'])

# Khởi tạo VietOCR
vietocr_config = Cfg.load_config_from_name('vgg_transformer')
vietocr_config['device'] = 'cpu'  # Hoặc 'cuda' nếu có GPU
vietocr_predictor = Predictor(vietocr_config)


def analyze_image(image):
    """Phân tích đặc điểm ảnh: độ sáng trung bình và độ tương phản."""
    mean_brightness = np.mean(image)
    contrast = np.std(image)
    return mean_brightness, contrast

def save_intermediate_step(image, step_name, save_path, step_index):
    """Lưu ảnh trung gian với tên có thứ tự."""
    filename = f"step_{step_index}_{step_name}.png"
    full_path = os.path.join(save_path, filename)
    cv2.imwrite(full_path, image)
    return full_path

def preprocess_image(image, intermediate_save_path=None):
    """Tiền xử lý ảnh với pipeline tối ưu hóa và lưu các bước trung gian."""
    intermediate_files = []
    step_index = 0

    # Phân tích ảnh
    mean_brightness, contrast = analyze_image(image)
    
    # Chuyển đổi định dạng ảnh
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format")
    
    if intermediate_save_path:
        intermediate_files.append(save_intermediate_step(gray, "grayscale", intermediate_save_path, step_index))
    step_index += 1

    # Tăng độ tương phản nếu cần (contrast < 50)
    if contrast < 50:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        if intermediate_save_path:
            intermediate_files.append(save_intermediate_step(gray, "clahe", intermediate_save_path, step_index))
        step_index += 1

    # Giảm nhiễu nếu ảnh quá sáng hoặc quá tối
    if mean_brightness > 200 or mean_brightness < 50:
        gray = cv2.fastNlMeansDenoising(gray)
        if intermediate_save_path:
            intermediate_files.append(save_intermediate_step(gray, "denoised", intermediate_save_path, step_index))
        step_index += 1

    # Chỉ áp dụng Gaussian blur nếu ảnh có độ phân giải cao (>1000px)
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if intermediate_save_path:
            intermediate_files.append(save_intermediate_step(gray, "blurred", intermediate_save_path, step_index))
        step_index += 1

    # Adaptive thresholding với tham số động
    block_size = 11 if image.shape[0] < 500 else 21  # Tăng block_size cho ảnh lớn
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, 2)
    if intermediate_save_path:
        intermediate_files.append(save_intermediate_step(thresh, "thresholded", intermediate_save_path, step_index))
    step_index += 1

    # Resize dựa trên kích thước ảnh gốc
    scale_percent = 200 if image.shape[0] < 500 else 100
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_AREA)
    if intermediate_save_path:
        intermediate_files.append(save_intermediate_step(resized, "resized", intermediate_save_path, step_index))

    return resized, intermediate_files

def preprocess_image_for_easyocr(image):
    """Chuẩn bị ảnh cho EasyOCR, đảm bảo định dạng RGB."""
    # Resize trước nếu ảnh quá lớn để tăng tốc độ
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = min(2000/image.shape[0], 2000/image.shape[1])
        new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Chuyển đổi định dạng
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
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
            
            # Tiền xử lý ảnh cho báo cáo (với lưu kết quả trung gian)
            processed_for_ocr, intermediate_files = preprocess_image(
                img_np, intermediate_save_path=app.config['INTERMEDIATE_FOLDER']
            )
            
            # Thực hiện OCR với EasyOCR
            result = reader.readtext(processed_img, detail=0, paragraph=True)
            text = '\n'.join(result)
            
            # Tạo tên file duy nhất
            filename = f"processed_{file.filename}"
            processed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Lưu ảnh đã xử lý
            cv2.imwrite(processed_img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            
            # Tạo URL cho ảnh và các bước trung gian
            processed_image_url = url_for('static', filename=f'uploads/{filename}')
            intermediate_urls = [
                url_for('static', filename=f'intermediate/{os.path.basename(f)}')
                for f in intermediate_files
            ]
            
            return jsonify({
                'text': text,
                'processed_image': processed_image_url,
                'intermediate_images': intermediate_urls
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 