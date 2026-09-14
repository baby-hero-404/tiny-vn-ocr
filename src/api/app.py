"""FastAPI REST API web service for Tiny VN OCR."""

from typing import Optional, List
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse

from src.pipeline import DocumentOCRPipeline
from src.schemas.document import DocumentResponse

app = FastAPI(
    title="Tiny VN OCR API",
    description="Lightweight Vietnamese identity document OCR web service",
    version="0.2.0",
)

pipeline = DocumentOCRPipeline()


@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint returning system status and available OCR engines."""
    return {
        "status": "healthy",
        "engines": pipeline.ocr_engine.available_engines,
    }


@app.post("/api/v1/ocr", response_model=DocumentResponse, summary="Perform Document OCR")
async def process_document_ocr(
    file: Optional[UploadFile] = File(None),
    front_file: Optional[UploadFile] = File(None),
    back_file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    document_type: str = Query("cccd", description="Document type: cccd, driving_license, vehicle_registration"),
):
    """Process uploaded document image(s) and return extracted structured JSON fields.

    Supports:
    - Single image: Pass via 'file' or 'front_file'
    - Dual images: Pass via 'front_file' and 'back_file', or multiple files via 'files'
    """
    uploaded_files: List[UploadFile] = []

    # Priority 1: Named dual files (front_file, back_file)
    if front_file is not None or back_file is not None:
        if front_file is not None:
            uploaded_files.append(front_file)
        if back_file is not None:
            uploaded_files.append(back_file)
    # Priority 2: Multi-file list (files)
    elif files is not None and len(files) > 0:
        uploaded_files.extend(files)
    # Priority 3: Legacy single file parameter (file)
    elif file is not None:
        uploaded_files.append(file)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Please provide 'file', 'front_file'/'back_file', or 'files'.",
        )

    decoded_images: List[np.ndarray] = []
    for f in uploaded_files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{f.filename or 'upload'}' must be a valid image format (e.g. image/jpeg, image/png).",
            )

        contents = await f.read()
        if not contents:
            continue

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            decoded_images.append(img)

    if not decoded_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode any valid image files.",
        )

    try:
        if len(decoded_images) >= 2:
            return pipeline.process_dual(decoded_images[0], decoded_images[1], document_type)
        else:
            return pipeline.process(decoded_images[0], document_type)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal OCR processing error: {str(e)}",
        )
