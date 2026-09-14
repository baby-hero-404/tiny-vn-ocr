# A quick mapping to prove the concept
PROVINCE_CODES = {
    "Sóc Trăng": "094",
    "Hà Nội": "001",
    "TP Hồ Chí Minh": "079",
    "An Giang": "089",
    "Cần Thơ": "092"
}

def reconstruct_prefix(province, gender, yob):
    prov_code = PROVINCE_CODES.get(province, "???")
    
    yob_int = int(yob)
    if 1900 <= yob_int <= 1999:
        g_code = "0" if gender == "Nam" else "1"
    elif 2000 <= yob_int <= 2099:
        g_code = "2" if gender == "Nam" else "3"
    else:
        g_code = "?"
        
    yob_code = str(yob)[-2:]
    
    return f"{prov_code}{g_code}{yob_code}"

print("Reconstructed for Sóc Trăng, Nữ, 2004:", reconstruct_prefix("Sóc Trăng", "Nữ", 2004))
