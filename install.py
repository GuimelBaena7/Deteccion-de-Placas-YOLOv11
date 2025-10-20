#!/usr/bin/env python3
"""
Script de instalación para el proyecto de detección de placas con YOLOv11
"""

import subprocess
import sys
import os

def install_requirements():
    """Instala las dependencias necesarias"""
    print("Instalando dependencias...")
    
    # Instalar dependencias básicas
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Descargar modelo YOLOv11 si no existe
    if not os.path.exists("yolo11n.pt"):
        print("Descargando modelo YOLOv11...")
        from ultralytics import YOLO
        model = YOLO('yolo11n.pt')  # Esto descargará automáticamente el modelo
        print("Modelo YOLOv11 descargado exitosamente")
    
    print("Instalación completada!")

if __name__ == "__main__":
    install_requirements()