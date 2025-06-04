import cv2
import numpy as np
from utils import is_blurry
import os

# Parámetros configurables
MIN_FACE_SIZE = 112       # Mínimo tamaño de rostro en px para ser considerado válido
BLUR_THRESHOLD = 100.0    # Umbral de varianza Laplaciana para evaluar enfoque

def validate_image(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {"valid": False, "reason": "Ruta no encontrada"}

    image = cv2.imread(image_path)
    if image is None:
        return {"valid": False, "reason": "Archivo corrupto o no es una imagen"}

    if is_blurry(image):
        return {"valid": False, "reason": "Imagen borrosa. Asegúrate de que esté bien enfocada."}

    height, width = image.shape[:2]
    if min(height, width) < MIN_FACE_SIZE:
        return {"valid": False, "reason": f"Imagen demasiado pequeña (< {MIN_FACE_SIZE}px)."}

    return {"valid": True, "image": image}

def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

def crop_and_align(image, bbox, margin=10):
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return image[y1:y2, x1:x2]
