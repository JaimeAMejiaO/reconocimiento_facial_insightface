import cv2
import os
import time
from tkinter import Tk, filedialog

# Carpeta donde se almacenan las imágenes capturadas o cargadas
DATA_DIR = os.path.join("data", "input")
os.makedirs(DATA_DIR, exist_ok=True)

def _build_filename(label: str) -> str:
    timestamp = int(time.time())
    return os.path.join(DATA_DIR, f"{label}_{timestamp}.jpg")

def capture_from_webcam(label: str, camera_id: int = 0) -> str:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo acceder a la cámara (ID: {camera_id})")

    print(f"[INFO] Usando cámara {camera_id}. Presiona 'c' para capturar o 'q' para cancelar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo capturar la imagen.")
            break

        cv2.imshow("Vista previa", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            path = _build_filename(label)
            cv2.imwrite(path, frame)
            print(f"[OK] Imagen guardada en {path}")
            break
        elif key == ord('q'):
            print("[CANCELADO] Captura cancelada por el usuario.")
            path = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return path

def load_image_from_file(label: str) -> str:
    root = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)

    filepath = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.webp")]
    )

    if not filepath:
        print("[CANCELADO] No se seleccionó ninguna imagen.")
        return None

    # Si el archivo ya está en la carpeta destino, no copiar
    abs_data_dir = os.path.abspath(DATA_DIR)
    abs_filepath = os.path.abspath(filepath)

    if os.path.commonpath([abs_filepath, abs_data_dir]) == abs_data_dir:
        print(f"[INFO] Imagen ya está en {DATA_DIR}, no se copia.")
        return abs_filepath

    # Si no está en la carpeta destino, copiarlo
    image = cv2.imread(filepath)
    if image is None:
        print("[ERROR] El archivo no es una imagen válida.")
        return None

    path = _build_filename(label)
    cv2.imwrite(path, image)
    print(f"[OK] Imagen copiada a {path}")
    return path
