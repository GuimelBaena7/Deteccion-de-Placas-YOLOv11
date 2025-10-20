# Detección de Placas con YOLOv11

Sistema de detección y reconocimiento de placas vehiculares utilizando YOLOv11 y OCR.

## Características

- Detección de vehículos usando YOLOv11
- Detección de placas vehiculares con modelo personalizado
- Seguimiento de vehículos con algoritmo SORT
- Reconocimiento de texto OCR con EasyOCR
- Interpolación de datos faltantes
- Visualización de resultados

## Instalación

### Opción 1: Instalación automática
```bash
python install.py
```

### Opción 2: Instalación manual
```bash
pip install -r requirements.txt
```

## Uso

### 1. Detección básica
```bash
python main.py
```

### 2. Interpolación de datos
```bash
python add_missing_data.py
```

### 3. Visualización
```bash
python visualize.py
```

## Archivos principales

- `main.py`: Script principal de detección
- `util.py`: Funciones utilitarias para OCR y procesamiento
- `visualize.py`: Visualización de resultados
- `add_missing_data.py`: Interpolación de datos faltantes
- `requirements.txt`: Dependencias del proyecto

## Modelos requeridos

- `yolo11n.pt`: Modelo YOLOv11 para detección de vehículos (se descarga automáticamente)
- `license_plate_detector.pt`: Modelo personalizado para detección de placas

## Formato de salida

El sistema genera un archivo CSV con las siguientes columnas:
- `frame_nmr`: Número de frame
- `car_id`: ID del vehículo
- `car_bbox`: Coordenadas del vehículo
- `license_plate_bbox`: Coordenadas de la placa
- `license_plate_bbox_score`: Confianza de detección de placa
- `license_number`: Texto de la placa
- `license_number_score`: Confianza del OCR

## Compatibilidad

- Python 3.8+
- YOLOv11 (ultralytics >= 8.3.0)
- OpenCV
- EasyOCR