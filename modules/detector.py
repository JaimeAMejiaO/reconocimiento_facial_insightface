from insightface.app import FaceAnalysis
import cv2
import os

# Inicializar RetinaFace al cargar el módulo
model_name = "buffalo_l"  # usa el modelo base de RetinaFace + ArcFace
model_path = os.path.join("models")

face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def detect_faces(image):
    faces = face_app.get(image)
    if not faces:
        return None
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in largest_face.bbox]
    cropped = image[y1:y2, x1:x2]
    return cropped

def get_largest_face(faces):
    """
    De una lista de detecciones, retorna la de mayor tamaño.
    """
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
