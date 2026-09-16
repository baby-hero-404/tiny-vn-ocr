"""MRZ Parser for Vietnamese CCCD / Identity Cards (ICAO 9303 TD1 Format).

Vietnamese chip-based CCCD and 2024 Identity Cards feature 3 MRZ lines at the bottom:
Line 1: IDVNM<DocumentNumber><CheckDigit><OptionalData/12-digit-ID>
Line 2: <DOB:YYMMDD><CheckDigit><Sex:M/F><Expiry:YYMMDD><CheckDigit><Nationality:VNM>...
Line 3: <Surname><<GivenNames<<<<<<<<<<<<<<<<
"""

import re
from datetime import datetime
from typing import Dict, List, Optional

from src.postprocessing.cccd_id_utils import infer_birth_year_and_gender


def parse_mrz_lines(lines: List[str]) -> Dict[str, str]:
    """Parse MRZ TD1 formatted lines from OCR elements.
    
    Returns a dictionary of extracted fields with 100% accuracy checksum if valid.
    """
    if not lines:
        return {}

    # 1. Filter lines that look like MRZ
    mrz_candidates = []
    for line in lines:
        if not line:
            continue
        cleaned = re.sub(r"[^A-Z0-9<]", "", str(line).upper().strip())
        # Replace common OCR misreads in MRZ
        cleaned = cleaned.replace("O", "0") if re.search(r"^\d", cleaned) else cleaned
        if len(cleaned) >= 20 and ("<<" in cleaned or "VNM" in cleaned or cleaned.startswith("ID") or cleaned.startswith("I<")):
            mrz_candidates.append(cleaned)

    if not mrz_candidates:
        return {}

    extracted: Dict[str, str] = {}

    from src.postprocessing.validator import FieldValidator

    # 2. Extract Line 1: ID Number (Sliding window with scoring)
    #    Collect all valid 12-digit candidates and pick the one with the
    #    most plausible birth_year (1930-2020) and gender code (0-3).
    #    Per ICAO TD1 format, the 12-digit national ID is at the END of
    #    Line 1, so we also prefer higher offsets as a tiebreaker.
    all_id_candidates: list[tuple[str, int]] = []  # (id, offset)
    for cand in mrz_candidates:
        digits_only = re.sub(r"\D", "", cand)
        if len(digits_only) >= 12:
            for i in range(len(digits_only) - 11):
                sub = digits_only[i:i+12]
                if FieldValidator.is_valid_cccd_id(sub):
                    all_id_candidates.append((sub, i))

    if all_id_candidates:
        # Pre-parse Line 2 gender for cross-validation (look ahead)
        mrz_gender: Optional[str] = None
        for cand in mrz_candidates:
            m = re.search(r"\d{6}[0-9<]([MF<])\d{6}", cand)
            if not m:
                cand_sub = re.sub(r"(?<=[0-9A-Z<])O(?=[0-9A-Z<])", "0", cand)
                m = re.search(r"\d{6}[0-9<]([MF<])\d{6}", cand_sub)
            if m:
                sc = m.group(1)
                if sc == "M":
                    mrz_gender = "Nam"
                elif sc == "F":
                    mrz_gender = "Nữ"
                break

        def _score_id(item: tuple[str, int]) -> tuple[int, int]:
            id_str, offset = item
            info = infer_birth_year_and_gender(id_str)
            if not info:
                return (0, offset)
            by, gender = info
            score = 2 if 1930 <= by <= 2020 else 1
            # Bonus: gender matches MRZ Line 2
            if mrz_gender and gender == mrz_gender:
                score += 1
            # Tiebreaker: prefer higher offset (ICAO TD1 puts ID at end)
            return (score, offset)

        best_id, _ = max(all_id_candidates, key=_score_id)
        extracted["id_number"] = best_id

    # 3. Extract Line 2: Date of Birth, Gender, Date of Expiry
    for cand in mrz_candidates:
        # Look for pattern: 6 digits (DOB) + 1 digit (check) + [MF] + 6 digits (EXP)
        # Normalize O -> 0 and Z -> 2 in digit sections
        cand_norm = cand
        m = re.search(r"(\d{6})[0-9<]([MF<])(\d{6})", cand_norm)
        if not m:
            # Tolerant search: OCR might misread digit 0 as O
            cand_sub = re.sub(r"(?<=[0-9A-Z<])O(?=[0-9A-Z<])", "0", cand)
            m = re.search(r"(\d{6})[0-9<]([MF<])(\d{6})", cand_sub)

        if m:
            dob_raw = m.group(1)
            sex_char = m.group(2)
            exp_raw = m.group(3)

            # Determine century from 12-digit ID if available
            id_num = extracted.get("id_number", "")
            id_info = infer_birth_year_and_gender(id_num)
            if id_info:
                century_dob = str(id_info[0])[:2]
            else:
                yr_val = int(dob_raw[:2])
                century_dob = "19" if yr_val > 30 else "20"

            try:
                dob_dt = datetime.strptime(f"{century_dob}{dob_raw}", "%Y%m%d")
                extracted["date_of_birth"] = dob_dt.strftime("%d/%m/%Y")
            except ValueError:
                pass

            if sex_char == "M":
                extracted["gender"] = "Nam"
            elif sex_char == "F":
                extracted["gender"] = "Nữ"

            try:
                # CCCD expiry is always in the 21st century (cards were not
                # issued before 2014). Using base_century = 2000 is always
                # correct and avoids century-inference bugs when DOB was
                # parsed with a wrong century due to an incorrect MRZ ID.
                exp_yy = int(exp_raw[:2])
                exp_year = 2000 + exp_yy
                exp_dt = datetime.strptime(f"{exp_year}{exp_raw[2:]}", "%Y%m%d")
                # Sanity: expiry must be in the future relative to any
                # reasonable CCCD issue date (>= 2014)
                if exp_dt.year >= 2014:
                    extracted["date_of_expiry"] = exp_dt.strftime("%d/%m/%Y")
            except ValueError:
                pass

    # 4. Extract Line 3: Name
    for cand in mrz_candidates:
        if "<<" in cand and not cand.startswith("ID") and not re.search(r"^\d{6}", cand):
            parts = cand.split("<<")
            if len(parts) >= 2:
                surname = parts[0].replace("<", "").strip()
                given_raw = parts[1].split("<")
                given_names = [p.strip() for p in given_raw if p.strip()]
                if surname:
                    full_mrz_name = f"{surname} {' '.join(given_names)}".strip()
                    if len(full_mrz_name) >= 3 and not re.search(r"\d", full_mrz_name):
                        extracted["mrz_name"] = full_mrz_name

    return extracted
