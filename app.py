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
from typing import Dict, List, Tuple, Optional
import json

app = Flask(__name__)

# Cấu hình đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Cấu hình thư mục
UPLOAD_FOLDER = 'static/uploads'
INTERMEDIATE_FOLDER = 'static/intermediate'
RESULTS_FOLDER = 'results'
for folder in [UPLOAD_FOLDER, INTERMEDIATE_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INTERMEDIATE_FOLDER'] = INTERMEDIATE_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Khởi tạo EasyOCR
reader = easyocr.Reader(['vi', 'en'])

# Khởi tạo VietOCR
vietocr_config = Cfg.load_config_from_name('vgg_transformer')
vietocr_config['device'] = 'cpu'  # Hoặc 'cuda' nếu có GPU
vietocr_predictor = Predictor(vietocr_config)

def perform_easyocr(image: np.ndarray, config: Dict) -> str:
    """
    Thực hiện OCR với EasyOCR với các tham số tùy chỉnh.
    
    Args:
        image: Ảnh đầu vào dạng numpy array
        config: Dict chứa các tham số cấu hình
        
    Returns:
        str: Văn bản được nhận dạng
    """
    result = reader.readtext(
        image,
        detail=0,
        paragraph=config.get('paragraph', True),
        text_threshold=config.get('text_threshold', 0.5),
        low_text=config.get('low_text', 0.4),
        decoder=config.get('decoder', 'greedy')
    )
    return '\n'.join(result)

def perform_tesseract(image: np.ndarray) -> str:
    """
    Thực hiện OCR với Tesseract.
    
    Args:
        image: Ảnh đầu vào dạng numpy array
        
    Returns:
        str: Văn bản được nhận dạng
    """
    return pytesseract.image_to_string(image, lang='vie+eng')

def perform_vietocr(image: np.ndarray) -> str:
    """
    Thực hiện OCR với VietOCR.
    
    Args:
        image: Ảnh đầu vào dạng numpy array
        
    Returns:
        str: Văn bản được nhận dạng
    """
    pil_image = Image.fromarray(image)
    return vietocr_predictor.predict(pil_image)

def evaluate_ocr(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Đánh giá kết quả OCR so với ground truth.
    
    Args:
        predicted: Văn bản được dự đoán
        ground_truth: Văn bản thực tế
        
    Returns:
        Dict chứa CER và WER
    """
    measures = jiwer.compute_measures(ground_truth, predicted)
    return {
        'cer': measures['substitutions'] + measures['deletions'] + measures['insertions'],
        'wer': measures['wer']
    }

def hybrid_ocr(image: np.ndarray, ground_truth: Optional[str] = None) -> Dict:
    """
    Thực hiện OCR với nhiều mô hình và chọn kết quả tốt nhất.
    
    Args:
        image: Ảnh đầu vào
        ground_truth: Văn bản thực tế (tùy chọn)
        
    Returns:
        Dict chứa kết quả của tất cả mô hình và đánh giá
    """
    # Cấu hình EasyOCR
    easyocr_configs = [
        {'text_threshold': 0.5, 'low_text': 0.4, 'paragraph': True, 'decoder': 'greedy'},
        {'text_threshold': 0.7, 'low_text': 0.4, 'paragraph': True, 'decoder': 'greedy'},
        {'text_threshold': 0.9, 'low_text': 0.4, 'paragraph': True, 'decoder': 'greedy'},
        {'text_threshold': 0.7, 'low_text': 0.4, 'paragraph': False, 'decoder': 'beamsearch'}
    ]
    
    # Thực hiện OCR với các mô hình
    results = {
        'tesseract': perform_tesseract(image),
        'vietocr': perform_vietocr(image),
        'easyocr': {}
    }
    
    # Thử nghiệm các cấu hình EasyOCR
    for i, config in enumerate(easyocr_configs):
        config_name = f'config_{i+1}'
        results['easyocr'][config_name] = perform_easyocr(image, config)
    
    # Đánh giá nếu có ground truth
    evaluation = None
    if ground_truth:
        evaluation = {
            'tesseract': evaluate_ocr(results['tesseract'], ground_truth),
            'vietocr': evaluate_ocr(results['vietocr'], ground_truth),
            'easyocr': {}
        }
        
        for config_name, text in results['easyocr'].items():
            evaluation['easyocr'][config_name] = evaluate_ocr(text, ground_truth)
    
    # Chọn kết quả tốt nhất (dựa trên độ dài văn bản)
    all_texts = [results['tesseract'], results['vietocr']] + list(results['easyocr'].values())
    best_text = max(all_texts, key=len)
    
    return {
        'best_text': best_text,
        'all_results': results,
        'evaluation': evaluation
    }

def test_ocr_models(image: np.ndarray, ground_truth: str) -> None:
    """
    Thử nghiệm các mô hình OCR và lưu kết quả vào CSV.
    
    Args:
        image: Ảnh đầu vào
        ground_truth: Văn bản thực tế
    """
    try:
        results = hybrid_ocr(image, ground_truth)
        
        # Chuẩn bị dữ liệu cho CSV
        data = []
        for model, text in results['all_results'].items():
            if model == 'easyocr':
                for config_name, config_text in text.items():
                    eval_data = results['evaluation'][model][config_name]
                    data.append({
                        'model': f'easyocr_{config_name}',
                        'text': config_text,
                        'cer': eval_data['cer'],
                        'wer': eval_data['wer'],
                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            else:
                eval_data = results['evaluation'][model]
                data.append({
                    'model': model,
                    'text': text,
                    'cer': eval_data['cer'],
                    'wer': eval_data['wer'],
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Tạo DataFrame
        df = pd.DataFrame(data)
        
        # Đảm bảo thư mục results tồn tại
        results_dir = app.config['RESULTS_FOLDER']
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Đường dẫn file CSV
        csv_path = os.path.join(results_dir, 'ocr_results.csv')
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(csv_path):
            # Nếu file đã tồn tại, thêm dữ liệu mới
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # Nếu file chưa tồn tại, tạo file mới với header
            df.to_csv(csv_path, mode='w', header=True, index=False)
            
        print(f"Đã lưu kết quả vào {csv_path}")
    except Exception as e:
        print(f"Lỗi khi lưu kết quả: {str(e)}")
        raise

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
            
            # Tiền xử lý ảnh cho OCR
            processed_img = preprocess_image_for_easyocr(img_np)
            
            # Tiền xử lý ảnh cho báo cáo (với lưu kết quả trung gian)
            processed_for_ocr, intermediate_files = preprocess_image(
                img_np, intermediate_save_path=app.config['INTERMEDIATE_FOLDER']
            )
            
            # Lấy ground truth nếu có
            ground_truth = request.form.get('ground_truth')
            
            # Thực hiện hybrid OCR
            ocr_results = hybrid_ocr(processed_img, ground_truth)
            
            # Lưu kết quả vào CSV nếu có ground truth
            if ground_truth:
                test_ocr_models(processed_img, ground_truth)
            
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
                'best_text': ocr_results['best_text'],
                'all_results': ocr_results['all_results'],
                'evaluation': ocr_results['evaluation'],
                'processed_image': processed_image_url,
                'intermediate_images': intermediate_urls
            })
        except Exception as e:
            return jsonify({'error': str(e)})
        

if __name__ == '__main__':
    app.run(debug=True) 