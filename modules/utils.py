import os
import cv2
import csv
import numpy as np

LOG_DIR = os.path.join("output", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "resultados.csv")
os.makedirs(LOG_DIR, exist_ok=True)

def is_blurry(image_path, threshold=35.0, is_document=False):
    """
    Evalúa si la imagen es borrosa usando la varianza del Laplaciano.
    Si es un documento, aplica un umbral más permisivo.
    """
    if is_document:
        threshold = 15.0  # Más permisivo para documentos

    image = cv2.imread(image_path)
    if image is None:
        return True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"[DEBUG] Variance of Laplacian for {image_path}: {variance:.2f}")
    return variance < threshold

def get_largest_face_bbox(faces):
    """
    Devuelve el rostro más grande basado en el área del bounding box.
    """
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

def is_face_size_valid(bbox, min_size=112):
    """
    Verifica que el ancho y alto del rostro sean al menos de min_size.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width >= min_size and height >= min_size

def rotate_image(image, angle):
    """
    Rota una imagen un número dado de grados.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def log_result(image1_path, image2_path, similarity, matched):
    header = ['Rostro', 'Documento', 'Similitud', 'Match']
    data = [image1_path, image2_path, f"{similarity:.4f}", "MATCH" if matched else "NO MATCH"]

    # Escribir encabezado si el archivo no existe
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)

    print(f"[LOG] Resultado registrado: {data}")

def log_result_dual(image1_path, image2_path, sim_arcface, match_arcface, sim_landmarks, match_landmarks):
    """
    Guarda los resultados de similitud ArcFace y con landmarks en un CSV.
    """
    header = [
        'Imagen Rostro', 'Imagen Documento',
        'Similitud ArcFace', 'Match ArcFace',
        'Similitud Landmarks', 'Match Landmarks'
    ]
    
    data = [
        image1_path,
        image2_path,
        f"{sim_arcface:.4f}",
        "MATCH" if match_arcface else "NO MATCH",
        f"{sim_landmarks:.4f}",
        "MATCH" if match_landmarks else "NO MATCH"
    ]

    # Escribir encabezado si el archivo aún no existe
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)

    print(f"[LOG] Resultado dual registrado: {data}")