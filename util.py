import string
import easyocr
import cv2
import numpy as np
import re

# 🔹 Inicializar EasyOCR con soporte en inglés y español
reader = easyocr.Reader(['en', 'es'], gpu=True)

# 🔹 Mapeos de caracteres comunes en placas colombianas
dict_char_to_int = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'G': '6', 'Z': '2'}
dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

# ----------------------------------------------------------
# 🧩 Guardar CSV de resultados
# ----------------------------------------------------------
def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                data = results[frame_nmr][car_id]

                if 'car' in data and 'license_plate' in data:
                    lp = data['license_plate']
                    car_bbox = [float(x) if isinstance(x, (int, float, np.floating)) else 0 for x in data['car']['bbox']]
                    lp_bbox = [float(x) if isinstance(x, (int, float, np.floating)) else 0 for x in lp['bbox']]
                    lp_bbox_score = float(lp['bbox_score']) if isinstance(lp['bbox_score'], (int, float, np.floating)) else 0
                    lp_text_score = float(lp['text_score']) if isinstance(lp['text_score'], (int, float, np.floating)) else 0

# ----------------------------------------------------------
# 🧩 Mejorada: Validación flexible para placas colombianas
# ----------------------------------------------------------
def license_complies_format(text):
    """
    Comprueba si el texto tiene el formato de una placa colombiana:
    - 3 letras + 3 números (ABC123)
    - 3 letras + 2 números + 1 letra (ABC12D)
    """
    pattern1 = r'^[A-Z]{3}[0-9]{3}$'      # ABC123
    pattern2 = r'^[A-Z]{3}[0-9]{2}[A-Z]$' # ABC12D
    return re.match(pattern1, text) or re.match(pattern2, text)

# ----------------------------------------------------------
# 🧩 Convertir letras/números parecidos
# ----------------------------------------------------------
def format_license(text):
    text = text.upper().replace(' ', '').replace('-', '')
    formatted = ''
    for char in text:
        if char in dict_char_to_int:
            formatted += dict_char_to_int[char]
        elif char in dict_int_to_char:
            formatted += dict_int_to_char[char]
        else:
            formatted += char
    return formatted

# ----------------------------------------------------------
# 🧩 Preprocesamiento avanzado antes de OCR
# ----------------------------------------------------------
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Mejorar contraste
    gray = cv2.equalizeHist(gray)

    # Filtro bilateral (reduce ruido sin borrar bordes)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 15)
    return thresh

# ----------------------------------------------------------
# 🧩 Lectura OCR robusta
# ----------------------------------------------------------
def read_license_plate(license_plate_crop):
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None

    try:
        # Preprocesar imagen antes del OCR
        plate_preprocessed = preprocess_plate(license_plate_crop)

        # Leer texto con EasyOCR
        detections = reader.readtext(plate_preprocessed)

        best_text = None
        best_score = 0

        for detection in detections:
            _, text, score = detection
            text = text.upper().replace(' ', '').replace('-', '')

            if license_complies_format(text):
                formatted = format_license(text)
                if score > best_score:
                    best_text, best_score = formatted, score

        if best_text:
            return best_text, best_score

    except Exception as e:
        print(f"⚠️ Error en OCR: {e}")
        return None, None

    return None, None

# ----------------------------------------------------------
# 🧩 Asignar placa a vehículo
# ----------------------------------------------------------
def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Si la placa está dentro de la caja del carro
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]

    return -1, -1, -1, -1, -1
