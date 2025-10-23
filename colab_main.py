# DETECTOR DE PLACAS VEHICULARES - VERSIÓN GOOGLE COLAB
# =====================================================

# Instalar dependencias
!pip install ultralytics>=8.3.0 pandas opencv-python numpy scipy easyocr filterpy

# Importar librerías
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import string
from google.colab import files
from google.colab import drive
import os

# Montar Google Drive (opcional)
drive.mount('/content/drive')

# Clase principal para detección de placas
class LicensePlateDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
        self.dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
        
    def license_complies_format(self, text):
        if len(text) != 7:
            return False
        return (text[0] in string.ascii_uppercase or text[0] in self.dict_int_to_char.keys()) and \
               (text[1] in string.ascii_uppercase or text[1] in self.dict_int_to_char.keys()) and \
               (text[2] in '0123456789' or text[2] in self.dict_char_to_int.keys()) and \
               (text[3] in '0123456789' or text[3] in self.dict_char_to_int.keys()) and \
               (text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char.keys()) and \
               (text[5] in string.ascii_uppercase or text[5] in self.dict_int_to_char.keys()) and \
               (text[6] in string.ascii_uppercase or text[6] in self.dict_int_to_char.keys())
    
    def format_license(self, text):
        license_plate_ = ''
        mapping = {0: self.dict_int_to_char, 1: self.dict_int_to_char, 4: self.dict_int_to_char, 
                  5: self.dict_int_to_char, 6: self.dict_int_to_char, 2: self.dict_char_to_int, 3: self.dict_char_to_int}
        for j in range(7):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        return license_plate_
    
    def read_license_plate(self, license_plate_crop):
        if license_plate_crop is None or license_plate_crop.size == 0:
            return None, None
        try:
            detections = self.reader.readtext(license_plate_crop)
            for detection in detections:
                bbox, text, score = detection
                text = text.upper().replace(' ', '')
                if self.license_complies_format(text):
                    return self.format_license(text), score
        except Exception as e:
            print(f"Error en OCR: {e}")
        return None, None
    
    def get_car(self, license_plate, vehicle_track_ids):
        x1, y1, x2, y2, score, class_id = license_plate
        for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return vehicle_track_ids[j]
        return -1, -1, -1, -1, -1

# Función principal de detección
def detect_license_plates(video_path, license_model_path):
    detector = LicensePlateDetector()
    
    # Cargar modelos
    print("Cargando modelos YOLO...")
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO(license_model_path)
    
    # Cargar video
    print(f"Procesando video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    vehicles = [2, 3, 5, 7]  # IDs de vehículos en COCO
    results = {}
    
    frame_nmr = -1
    ret = True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        
        if ret:
            if frame_nmr % 50 == 0:  # Mostrar progreso cada 50 frames
                print(f"Procesando frame {frame_nmr}/{total_frames}")
                
            results[frame_nmr] = {}
            
            # Detectar vehículos
            detections = coco_model(frame)[0]
            detections_ = []
            
            if detections.boxes is not None:
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])
            
            # Simular tracking simple
            track_ids = [[*det, i] for i, det in enumerate(detections_)]
            
            # Detectar placas
            license_plates = license_plate_detector(frame)[0]
            
            if license_plates.boxes is not None:
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    
                    # Asignar placa a vehículo
                    xcar1, ycar1, xcar2, ycar2, car_id = detector.get_car(license_plate, track_ids)
                    
                    if car_id != -1:
                        # Recortar placa
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        
                        # Procesar placa
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        
                        # Leer texto de placa
                        license_plate_text, license_plate_text_score = detector.read_license_plate(license_plate_crop_thresh)
                        
                        if license_plate_text is not None:
                            results[frame_nmr][car_id] = {
                                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                'license_plate': {
                                    'bbox': [x1, y1, x2, y2],
                                    'text': license_plate_text,
                                    'bbox_score': score,
                                    'text_score': license_plate_text_score
                                }
                            }
                            print(f"Frame {frame_nmr}: Placa detectada: {license_plate_text} (confianza: {license_plate_text_score:.2f})")
    
    cap.release()
    print("Procesamiento completado!")
    return results

# Función para guardar resultados
def save_results(results, output_path):
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                data = results[frame_nmr][car_id]
                if 'car' in data and 'license_plate' in data and 'text' in data['license_plate']:
                    f.write(f"{frame_nmr},{car_id},"
                           f"[{' '.join(map(str, data['car']['bbox']))}],"
                           f"[{' '.join(map(str, data['license_plate']['bbox']))}],"
                           f"{data['license_plate']['bbox_score']},"
                           f"{data['license_plate']['text']},"
                           f"{data['license_plate']['text_score']}\n")

# EJECUCIÓN PRINCIPAL
print("=== DETECTOR DE PLACAS VEHICULARES ===")
print("Sube tu video y el modelo license_plate_detector.pt")

# Subir archivos
uploaded = files.upload()

# Verificar archivos subidos
video_file = None
model_file = None

for filename in uploaded.keys():
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_file = filename
        print(f"Video encontrado: {filename}")
    elif filename == 'license_plate_detector.pt':
        model_file = filename
        print(f"Modelo encontrado: {filename}")

if video_file and model_file:
    print(f"\nIniciando procesamiento...")
    print(f"Video: {video_file}")
    print(f"Modelo: {model_file}")
    
    # Ejecutar detección
    results = detect_license_plates(video_file, model_file)
    
    # Guardar resultados
    save_results(results, 'resultados_placas.csv')
    
    # Mostrar estadísticas
    total_detections = sum(len(frame_data) for frame_data in results.values())
    print(f"\n=== RESULTADOS ===")
    print(f"Total de detecciones: {total_detections}")
    print(f"Frames procesados: {len(results)}")
    
    # Mostrar algunas detecciones
    print("\nPrimeras detecciones:")
    count = 0
    for frame_nmr, frame_data in results.items():
        for car_id, data in frame_data.items():
            if 'license_plate' in data and 'text' in data['license_plate']:
                print(f"Frame {frame_nmr}: {data['license_plate']['text']}")
                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break
    
    print("\nDescargando archivo de resultados...")
    files.download('resultados_placas.csv')
    
else:
    print("❌ Error: Necesitas subir:")
    if not video_file:
        print("- Un archivo de video (.mp4, .avi, .mov, .mkv)")
    if not model_file:
        print("- El modelo license_plate_detector.pt")