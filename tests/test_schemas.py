"""Unit tests for document schemas."""

import pytest
from pydantic import ValidationError
from src.schemas.document import (
    CCCDFields,
    GPLXFields,
    VehicleRegistrationFields,
    DocumentResponse,
    GenderEnum,
    ProcessingMetadata,
)


def test_cccd_fields_valid():
    raw_data = {
        "id_number": "079123456789",
        "full_name": "NGUYEN VAN A",
        "date_of_birth": "01/01/1995",
        "gender": "NAM",
        "nationality": "VIET NAM",
        "place_of_origin": "TP HO CHI MINH",
        "place_of_residence": "QUAN 1 TP HO CHI MINH",
        "date_of_issue": "01/01/2022",
    }
    fields = CCCDFields(**raw_data)
    assert fields.id_number == "079123456789"
    assert fields.gender == GenderEnum.NAM

    resp = DocumentResponse(
        document_type="cccd",
        confidence=0.95,
        fields=fields.model_dump(),
        metadata=ProcessingMetadata(ocr_engine="rapidocr", processing_time_ms=120.5),
    )
    assert resp.document_type == "cccd"
    assert resp.fields["id_number"] == "079123456789"


def test_cccd_invalid_id_number():
    with pytest.raises(ValidationError):
        CCCDFields(id_number="12345")  # Less than 12 digits


def test_cccd_invalid_date():
    with pytest.raises(ValidationError):
        CCCDFields(date_of_birth="1995-01-01")  # Wrong format


def test_gplx_fields_valid():
    raw_data = {
        "license_number": "790123456789",
        "full_name": "NGUYEN VAN A",
        "date_of_birth": "01/01/1995",
        "nationality": "VIET NAM",
        "license_class": "A1",
        "date_of_issue": "10/05/2020",
    }
    fields = GPLXFields(**raw_data)
    assert fields.license_number == "790123456789"
    assert fields.license_class == "A1"

    resp = DocumentResponse(
        document_type="driving_license",
        confidence=0.92,
        fields=fields.model_dump(),
    )
    assert resp.document_type == "driving_license"


def test_vehicle_registration_fields_valid():
    raw_data = {
        "license_plate": "59A3-12345",
        "owner_name": "NGUYEN VAN A",
        "engine_number": "JF86E1234567",
        "chassis_number": "RLHJF5800NY123456",
        "date_of_issue": "01/01/2023",
    }
    fields = VehicleRegistrationFields(**raw_data)
    assert fields.license_plate == "59A3-12345"
    assert fields.engine_number == "JF86E1234567"

    resp = DocumentResponse(
        document_type="vehicle_registration",
        confidence=0.98,
        fields=fields.model_dump(),
    )
    assert resp.document_type == "vehicle_registration"
