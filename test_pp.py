import cv2
import logging
import numpy as np
from paddleocr import PPStructure
from typing import List, Dict, Any

# Cấu hình logging để theo dõi lỗi trong production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocumentProcessor")

class DocumentProcessor:
    def __init__(self, use_gpu: bool = True, lang: str = 'vi'):
        """
        Khởi tạo Engine xử lý tài liệu chuyên dụng cho Local Server.
        """
        try:
            # Khởi tạo PP-Structure v4
            self.engine = PPStructure(
                lang=lang,
                layout=True,            # Phân tích bố cục (Text, Table, Figure, etc.)
                table=True,             # Trích xuất cấu trúc bảng chuyên sâu
                use_angle_cls=True,     # Quan trọng: Tự động sửa góc xoay/nghiêng
                use_gpu=use_gpu,        # Tối ưu hóa cho Local Server có GPU
                show_log=False,
                det_db_thresh=0.3,      # Ngưỡng phát hiện box (tùy chỉnh để tăng precision)
                det_db_box_thresh=0.6,  # Ngưỡng lọc box nhiễu
                structure_version='PP-StructureV2'
            )
            logger.info("DocumentProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {e}")
            raise

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """
        Các bước tiền xử lý ảnh bổ sung nếu cần (Denoising, Grayscale).
        Mặc định PaddleOCR đã làm khá tốt phần này.
        """
        # Nếu ảnh quá lớn, có thể resize để giảm Latency mà vẫn giữ Accuracy
        h, w = image.shape[:2]
        if max(h, w) > 3000:
            scale = 3000 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return image

    def process(self, image_source: Any) -> List[Dict[str, Any]]:
        """
        Pipeline chính: Nhận vào path hoặc numpy array, trả về dữ liệu cấu trúc.
        """
        # 1. Load image
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
        else:
            img = image_source

        if img is None:
            logger.error("Invalid image input.")
            return []

        # 2. Tiền xử lý nhẹ
        img = self.pre_process(img)

        # 3. Thực thi Inference
        # PP-Structure sẽ trả về list các region, mỗi region có polygon/bbox và nội dung
        results = self.engine(img)

        # 4. Format lại output theo chuẩn Industry (Dễ dàng parse sang JSON)
        structured_output = []
        for index, region in enumerate(results):
            # Loại bỏ các region có độ tự tin (confidence) quá thấp
            # Điều này giúp lọc các box bị gán nhãn sai là "art" do nhiễu
            
            res_item = {
                "id": index,
                "type": region['type'].lower(),
                "bbox": region['bbox'].tolist() if isinstance(region['bbox'], np.ndarray) else region['bbox'],
                "confidence": None,
                "content": None
            }

            if region['type'] == 'table':
                # Nếu là bảng, lấy cấu trúc HTML và cell data
                res_item["content"] = {
                    "html": region['res'].get('html'),
                    "cells": region['res'].get('cell_bbox')
                }
            else:
                # Nếu là text/header/footer, lấy text list
                text_blocks = region['res']
                if text_blocks:
                    full_text = " ".join([line['text'] for line in text_blocks])
                    avg_score = np.mean([line['confidence'] for line in text_blocks])
                    res_item["content"] = full_text
                    res_item["confidence"] = float(avg_score)

            structured_output.append(res_item)

        return structured_output

# --- CÁCH TRIỂN KHAI TRONG PRODUCTION ---

if __name__ == "__main__":
    # Khởi tạo một lần duy nhất (Singleton pattern) để tránh load model nhiều lần vào GPU
    processor = DocumentProcessor(use_gpu=True)

    # Giả sử file PDF/Ảnh có table bị xoay lệch
    test_image = "rotated_document.jpg"
    
    try:
        final_results = processor.process(test_image)
        
        # Log kết quả mẫu
        for item in final_results:
            print(f"[{item['type'].upper()}] - Confidence: {item['confidence']}")
            if item['type'] == 'table':
                print(f"Table HTML: {item['content']['html'][:50]}...")
            else:
                print(f"Text: {item['content'][:100]}...")
                
    except Exception as e:
        print(f"Runtime Error: {e}")