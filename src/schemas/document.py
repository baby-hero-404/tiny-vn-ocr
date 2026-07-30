"""Document data schemas for Tiny VN OCR."""

from enum import Enum
from typing import Dict, Any, Optional
import re
from pydantic import BaseModel, Field, field_validator


class GenderEnum(str, Enum):
    NAM = "NAM"
    NU = "NU"


class LicenseClassEnum(str, Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A = "A"
    B1 = "B1"
    B2 = "B2"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    FB2 = "FB2"
    FC = "FC"
    FD = "FD"
    FE = "FE"


class CCCDFields(BaseModel):
    id_number: Optional[str] = Field(None, description="12-digit ID number")
    full_name: Optional[str] = Field(None, description="Full name")
    date_of_birth: Optional[str] = Field(None, description="Date of birth DD/MM/YYYY")
    gender: Optional[GenderEnum] = Field(None, description="Gender NAM/NU")
    nationality: Optional[str] = Field("VIET NAM", description="Nationality")
    place_of_origin: Optional[str] = Field(None, description="Place of origin")
    place_of_residence: Optional[str] = Field(None, description="Place of residence")
    date_of_issue: Optional[str] = Field(None, description="Date of issue DD/MM/YYYY")

    @field_validator("id_number")
    @classmethod
    def validate_id_number(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "":
            if not re.match(r"^\d{12}$", v):
                raise ValueError("ID number must be exactly 12 digits")
        return v

    @field_validator("date_of_birth", "date_of_issue")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "":
            if not re.match(r"^\d{2}/\d{2}/\d{4}$", v):
                raise ValueError("Date must be in DD/MM/YYYY format")
        return v


class GPLXFields(BaseModel):
    license_number: Optional[str] = Field(None, description="Driving license number")
    full_name: Optional[str] = Field(None, description="Full name")
    date_of_birth: Optional[str] = Field(None, description="Date of birth DD/MM/YYYY")
    nationality: Optional[str] = Field("VIET NAM", description="Nationality")
    license_class: Optional[str] = Field(None, description="License class e.g. A1, B2")
    date_of_issue: Optional[str] = Field(None, description="Date of issue DD/MM/YYYY")
    date_of_expiry: Optional[str] = Field(None, description="Date of expiry DD/MM/YYYY or null")

    @field_validator("date_of_birth", "date_of_issue", "date_of_expiry")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "":
            if not re.match(r"^\d{2}/\d{2}/\d{4}$", v):
                raise ValueError("Date must be in DD/MM/YYYY format")
        return v


class VehicleRegistrationFields(BaseModel):
    license_plate: Optional[str] = Field(None, description="License plate number")
    owner_name: Optional[str] = Field(None, description="Owner full name")
    owner_address: Optional[str] = Field(None, description="Owner address")
    brand: Optional[str] = Field(None, description="Vehicle brand")
    model: Optional[str] = Field(None, description="Vehicle model")
    color: Optional[str] = Field(None, description="Vehicle color")
    engine_number: Optional[str] = Field(None, description="Engine number")
    chassis_number: Optional[str] = Field(None, description="Chassis number")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    date_of_issue: Optional[str] = Field(None, description="Date of issue DD/MM/YYYY")

    @field_validator("date_of_issue")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "":
            if not re.match(r"^\d{2}/\d{2}/\d{4}$", v):
                raise ValueError("Date must be in DD/MM/YYYY format")
        return v


class ProcessingMetadata(BaseModel):
    ocr_engine: str = Field("rapidocr", description="OCR engine used")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")


class DocumentResponse(BaseModel):
    document_type: str = Field(..., description="Type of document (cccd, driving_license, vehicle_registration)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Overall confidence score")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Extracted field values")
    metadata: ProcessingMetadata = Field(default_factory=ProcessingMetadata, description="Execution metadata")
