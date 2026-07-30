# Tiny VN OCR

Hệ thống OCR trích xuất thông tin giấy tờ tùy thân (CCCD, GPLX, Giấy đăng ký xe) tối ưu trên thiết bị tài nguyên hạn chế (RAM ~2GB, CPU-only).

## Tính năng chính

- **Hỗ trợ đa loại giấy tờ**: Căn cước công dân (CCCD), Giấy phép lái xe (GPLX), Giấy đăng ký xe (Cà vẹt xe).
- **Template-based ROI Extraction**: Cắt chính xác từng vùng thông tin (ROI) để OCR thay vì OCR toàn bộ ảnh.
- **Tự động căn chỉnh ảnh (Image Alignment)**: Sử dụng thuật toán ORB Feature Detection và Homography RANSAC trong OpenCV để đưa ảnh về khung chuẩn.
- **Dual OCR Engine**: Tích hợp RapidOCR (ONNX Runtime CPU) làm engine chính và Tesseract làm fallback.
- **Hậu xử lý & Chuẩn hóa dữ liệu**: Tự động sửa lỗi nhận dạng ký tự (ví dụ `O` -> `0`, `S` -> `5`), kiểm tra định dạng Regex (Số CCCD 12 số, ngày tháng `DD/MM/YYYY`) và sửa lỗi bằng Fuzzy Matching (`rapidfuzz`).
- **REST API HTTP**: Cung cấp Web Service đơn giản xây dựng trên FastAPI.

## Cấu trúc thư mục

```
tiny-vn-ocr/
├── src/
│   ├── api/             # FastAPI REST endpoints
│   ├── ocr/             # Unified OCR Engine (RapidOCR & Tesseract)
│   ├── postprocessing/  # Chuẩn hóa văn bản, Regex & Fuzzy Matching
│   ├── preprocessing/   # Căn chỉnh ảnh (Homography) & Bộ lọc tối ưu OCR
│   ├── roi/             # Định nghĩa Template & Cắt vùng ROI
│   ├── schemas/         # Unified Pydantic Data Models
│   └── pipeline.py      # Quy trình xử lý Document OCR End-to-End
├── tests/               # Bộ kiểm thử Unit & Integration
├── pyproject.toml       # Cấu hình dự án
├── requirements.txt     # Các gói thư viện phụ thuộc
└── README.md
```

## Cài đặt & Sử dụng

### 1. Cài đặt môi trường

Yêu cầu Python >= 3.9. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

*(Tùy chọn)* Cài đặt Tesseract OCR nếu muốn sử dụng Tesseract làm engine fallback:
```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-vie
```

### 2. Khởi chạy REST API Service

Khởi chạy web server với Uvicorn:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. API Usage

#### Kiểm tra trạng thái hệ thống (Health Check)

```bash
curl -X GET http://localhost:8000/health
```

*Response:*
```json
{
  "status": "healthy",
  "engines": {
    "rapidocr": true,
    "tesseract": false
  }
}
```

#### Trích xuất thông tin giấy tờ (OCR Endpoint)

```bash
curl -X POST "http://localhost:8000/api/v1/ocr?document_type=cccd" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/cccd_image.jpg"
```

*Response (Unified JSON Schema):*
```json
{
  "document_type": "cccd",
  "confidence": 0.95,
  "fields": {
    "id_number": "079123456789",
    "full_name": "NGUYEN VAN A",
    "date_of_birth": "01/01/1995",
    "gender": "NAM",
    "nationality": "VIET NAM",
    "place_of_origin": "TP HO CHI MINH",
    "place_of_residence": "QUAN 1 TP HO CHI MINH",
    "date_of_issue": "01/01/2022"
  },
  "metadata": {
    "ocr_engine": "rapidocr",
    "processing_time_ms": 340.5
  }
}
```

các `document_type` hỗ trợ: `cccd`, `driving_license`, `vehicle_registration`.

### 4. Chạy kiểm thử (Testing)

Chạy bộ kiểm thử tự động với Pytest:

```bash
python -m pytest -v
```

## Kiến trúc hệ thống

```
Image Input -> Quality & Alignment (ORB Homography) -> ROI Extraction -> RapidOCR (Primary) / Tesseract (Fallback) -> Text Normalization & Regex/Fuzzy Validation -> Unified JSON
```

Xem thông tin chi tiết tại [docs/architecture.md](docs/architecture.md).
