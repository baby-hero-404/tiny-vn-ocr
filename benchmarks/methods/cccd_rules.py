"""CCCD Business Rules Engine — Cross-validation logic for Vietnamese ID cards."""

import re
from typing import Dict
from datetime import datetime


# CCCD expiry rules: cards expire at age 25, 40, 60, or never (issued after 60)
CCCD_EXPIRY_AGES = [25, 40, 60]


def validate_and_fix_expiry(fields: Dict[str, str]) -> Dict[str, str]:
    """Cross-validate date_of_expiry against date_of_birth using CCCD rules.
    
    CCCD always expires on the holder's birthday at age 25, 40, or 60.
    This means: expiry.day == dob.day AND expiry.month == dob.month.
    """
    dob_str = fields.get("date_of_birth", "")
    exp_str = fields.get("date_of_expiry", "")
    
    if not dob_str:
        return fields
    
    try:
        dob = datetime.strptime(dob_str, "%d/%m/%Y")
    except ValueError:
        return fields
    
    # Case 1: No expiry found — compute it from DOB
    if not exp_str:
        fields["date_of_expiry"] = _compute_expiry(dob)
        return fields
    
    try:
        exp = datetime.strptime(exp_str, "%d/%m/%Y")
    except ValueError:
        fields["date_of_expiry"] = _compute_expiry(dob)
        return fields
    
    # Case 2: Expiry == DOB (parser duplicated the date)
    if exp_str == dob_str:
        fields["date_of_expiry"] = _compute_expiry(dob)
        return fields
    
    # Case 3: Expiry year is before or equal to DOB year (clearly wrong)
    if exp.year <= dob.year:
        fields["date_of_expiry"] = _compute_expiry(dob)
        return fields
    
    # Case 4: Expiry age gap is not one of the valid milestones
    age_at_expiry = exp.year - dob.year
    if age_at_expiry not in CCCD_EXPIRY_AGES:
        # Try to find the closest valid milestone
        fields["date_of_expiry"] = _compute_expiry(dob, hint_year=exp.year)
        return fields
    
    # Case 5: Day/month don't match DOB (OCR misread month/day)
    if exp.day != dob.day or exp.month != dob.month:
        corrected = f"{dob.day:02d}/{dob.month:02d}/{exp.year}"
        # Re-validate the corrected year makes sense
        corrected_age = exp.year - dob.year
        if corrected_age in CCCD_EXPIRY_AGES:
            fields["date_of_expiry"] = corrected
        else:
            fields["date_of_expiry"] = _compute_expiry(dob)
        return fields
    
    return fields


def _compute_expiry(dob: datetime, hint_year: int = None) -> str:
    """Compute the most likely expiry date from DOB using CCCD rules."""
    current_year = datetime.now().year
    
    if hint_year:
        # Find the closest valid expiry age to the hint
        best_age = min(CCCD_EXPIRY_AGES, key=lambda a: abs((dob.year + a) - hint_year))
        exp_year = dob.year + best_age
    else:
        # Find the next future expiry milestone
        for age in CCCD_EXPIRY_AGES:
            exp_year = dob.year + age
            if exp_year >= current_year:
                break
        else:
            # Person is over 60, card has no expiry — use 60-year milestone
            exp_year = dob.year + 60
    
    return f"{dob.day:02d}/{dob.month:02d}/{exp_year}"


def fix_gender_positional(fields: Dict[str, str], lines: list) -> Dict[str, str]:
    """Fallback gender extraction using positional heuristic.
    
    On CCCD front, gender always appears near or after the DOB line.
    We look for standalone 'Nam' or 'Nữ' on lines near the DOB.
    """
    if fields.get("gender"):
        return fields
    
    # Find DOB line index
    dob_idx = -1
    for i, line in enumerate(lines):
        ll = line.lower()
        if "sinh" in ll or "birth" in ll:
            dob_idx = i
            break
    
    if dob_idx == -1:
        return fields
    
    # Search lines around DOB (typically DOB+1 to DOB+3)
    for i in range(max(0, dob_idx), min(len(lines), dob_idx + 5)):
        line = lines[i].strip()
        # Look for standalone gender words (not part of longer text)
        if re.match(r'^(Nam|nam|NAM)$', line):
            fields["gender"] = "Nam"
            return fields
        if re.match(r'^(Nữ|nữ|NỮ|NU|Nu)$', line):
            fields["gender"] = "Nữ"
            return fields
        # Also match within a line like "Giới tỉnh son\n24/09/2000\nNam"
        words = line.split()
        for w in words:
            clean_w = re.sub(r'[^a-zA-ZÀ-ỹ]', '', w)
            if clean_w.lower() == "nam" and len(words) <= 3:
                fields["gender"] = "Nam"
                return fields
            if clean_w.lower() in ("nữ", "nu") and len(words) <= 3:
                fields["gender"] = "Nữ"
                return fields
    
    return fields
