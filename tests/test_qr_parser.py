"""Unit tests for QR parser module."""

from src.postprocessing.qr_parser import parse_qr_payload, _format_date


def test_format_date():
    assert _format_date("20061990") == "20/06/1990"
    assert _format_date("26092025") == "26/09/2025"
    assert _format_date("invalid") == "invalid"


def test_parse_qr_payload_standard():
    payload = "094190004172||DIỆP THỊ THÚY NGUYỄN|20061990|Nữ|Ấp Đại Ân, Đại Tâm, Mỹ Xuyên, Sóc Trăng|20062020"
    fields = parse_qr_payload(payload)
    assert fields is not None
    assert fields["id_number"] == "094190004172"
    assert fields["full_name"] == "DIỆP THỊ THÚY NGUYỄN"
    assert fields["date_of_birth"] == "20/06/1990"
    assert fields["gender"] == "Nữ"
    assert fields["place_of_residence"] == "Ấp Đại Ân, Đại Tâm, Mỹ Xuyên, Sóc Trăng"
    assert fields["date_of_issue"] == "20/06/2020"


def test_parse_qr_payload_with_cmnd():
    payload = "094190004172|365998811|NGUYỄN VĂN A|01011995|Nam|Số 10 Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP Hồ Chí Minh|01012021"
    fields = parse_qr_payload(payload)
    assert fields is not None
    assert fields["id_number"] == "094190004172"
    assert fields["old_id_number"] == "365998811"
    assert fields["full_name"] == "NGUYỄN VĂN A"
    assert fields["gender"] == "Nam"


def test_parse_qr_payload_invalid():
    assert parse_qr_payload("") is None
    assert parse_qr_payload("short|payload") is None
    assert parse_qr_payload("plain text without pipes") is None
