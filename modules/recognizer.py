import numpy as np
import onnxruntime as ort
import cv2
import os

#Landmark
from insightface.app import FaceAnalysis

# Ruta al modelo ArcFace ResNet100
MODEL_PATH = os.path.join("models", "arcface_resnet100.onnx")

# Inicializar el modelo ONNX
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def preprocess(image):
    """
    Preprocesa la imagen: redimensiona, cambia a RGB, normaliza y reordena canales.
    """
    image = cv2.resize(image, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image -= 127.5
    image /= 128.0
    image = np.expand_dims(image, axis=0)
    return image

def get_embedding(face_img):
    """
    Obtiene el embedding facial a partir de una imagen de rostro recortado.
    """
    if face_img is None:
        return None

    input_blob = preprocess(face_img)
    outputs = session.run(None, {session.get_inputs()[0].name: input_blob})
    embedding = outputs[0][0]
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

def compare_faces(face1, face2, threshold=0.4):
    """
    Compara dos imágenes de rostros usando ArcFace y retorna si hay match y el score.
    """
    emb1 = get_embedding(face1)
    emb2 = get_embedding(face2)

    if emb1 is None or emb2 is None:
        print("[DEBUG] No se pudo extraer embeddings.")
        return False, 0.0

    score = np.dot(emb1, emb2)
    return score > threshold, score

# Modelo buffalo_l para prueba con landmarks (no afecta al principal)
buffalo_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
buffalo_app.prepare(ctx_id=0)

def compare_faces_with_landmarks(img1, img2, threshold=0.4):
    """
    Usa buffalo_l para extraer embeddings incluyendo landmarks (106 puntos) y compara.
    """
    faces1 = buffalo_app.get(img1)
    faces2 = buffalo_app.get(img2)

    if not faces1 or not faces2:
        print("[DEBUG] No se detectaron rostros (landmarks).")
        return False, 0.0

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    if emb1 is None or emb2 is None:
        print("[DEBUG] No se pudieron extraer embeddings con landmarks.")
        return False, 0.0

    score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return score > threshold, score