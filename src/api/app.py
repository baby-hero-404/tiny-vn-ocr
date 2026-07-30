"""FastAPI REST API web service for Tiny VN OCR."""

from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse

from src.pipeline import DocumentOCRPipeline
from src.schemas.document import DocumentResponse

app = FastAPI(
    title="Tiny VN OCR API",
    description="Lightweight Vietnamese identity document OCR web service",
    version="0.1.0",
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
    file: UploadFile = File(...),
    document_type: str = Query("cccd", description="Document type: cccd, driving_license, vehicle_registration"),
):
    """Process an uploaded document image and return extracted structured JSON fields."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided must be a valid image format (e.g. image/jpeg, image/png).",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Decode image bytes using OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image file.",
        )

    try:
        response = pipeline.process(image, document_type)
        return response
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
