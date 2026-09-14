with open("benchmarks/methods/layout_graph_parser.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "xmin, ymin, xmax, ymax = id_el[\"bbox\"]" in line:
        new_lines.extend([
            '                if "crop" in id_el:\n',
            '                    crop = id_el["crop"]\n',
            '                else:\n',
            '                    xmin, ymin, xmax, ymax = id_el["bbox"]\n',
            '                    pad_x, pad_y = 3, 5\n',
            '                    c_xmin = max(0, int(xmin) - pad_x)\n',
            '                    c_ymin = max(0, int(ymin) - pad_y)\n',
            '                    c_xmax = min(image.shape[1], int(xmax) + pad_x)\n',
            '                    c_ymax = min(image.shape[0], int(ymax) + pad_y)\n',
            '                    crop = image[c_ymin:c_ymax, c_xmin:c_xmax]\n',
            '\n',
            '                # Bước 2: Multi preprocessing\n',
            '                crop_orig = crop.copy()\n',
            '                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)\n',
            '                blurred = cv2.GaussianBlur(gray, (0, 0), 3)\n',
            '                sharpened_gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)\n',
            '                crop_sharp = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)\n',
            '                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\n',
            '                crop_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)\n',
            '\n',
            '                best_cand = cand\n',
            '                for c_img in [crop_orig, crop_sharp, crop_thresh]:\n',
            '                    res = ocr_engine.recognize_crop_vietocr(c_img)\n',
            '                    res_clean = re.sub(r"\\D", "", res)\n',
            '                    m = re.search(r"\\d{12}", res_clean)\n',
            '                    if m:\n',
            '                        val = m.group(0)\n',
            '                        if FieldValidator.is_valid_cccd_id(val):\n',
            '                            best_cand = val\n',
            '                            break\n',
            '                fields["id_number"] = best_cand\n'
        ])
        skip = True
    elif skip and "fields[\"id_number\"] = best_cand" in line:
        skip = False
    elif not skip:
        new_lines.append(line)

with open("benchmarks/methods/layout_graph_parser.py", "w") as f:
    f.writelines(new_lines)
