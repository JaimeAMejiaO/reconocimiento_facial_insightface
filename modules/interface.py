import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
from capture import capture_from_webcam, load_image_from_file
from detector import detect_faces
from recognizer import compare_faces, compare_faces_with_landmarks
from utils import is_blurry, log_result


# Variables globales para guardar rutas
img_rostro_path = None
img_doc_path = None


def capturar_rostro():
    global img_rostro_path
    img_rostro_path = capture_from_webcam("rostro", camera_id_var.get())
    lbl_estado_rostro.config(text=f"Rostro capturado")

def capturar_documento():
    global img_doc_path
    img_doc_path = capture_from_webcam("documento", camera_id_var.get())
    lbl_estado_doc.config(text=f"Documento capturado")

def cargar_rostro():
    global img_rostro_path
    img_rostro_path = load_image_from_file("rostro")
    if img_rostro_path:
        lbl_estado_rostro.config(text="Imagen cargada")

def cargar_documento():
    global img_doc_path
    img_doc_path = load_image_from_file("documento")
    if img_doc_path:
        lbl_estado_doc.config(text="Imagen cargada")

def comparar():
    if not img_rostro_path or not img_doc_path:
        messagebox.showwarning("Faltan imágenes", "Debes capturar o cargar ambas imágenes.")
        return

    if is_blurry(img_rostro_path, is_document=False):
        messagebox.showwarning("Rostro borroso", "La imagen del rostro está borrosa.")
        return
    if is_blurry(img_doc_path, is_document=True):
        messagebox.showwarning("Documento borroso", "La imagen del documento está borrosa.")
        return

    rostro_img = cv2.imread(img_rostro_path)
    doc_img = cv2.imread(img_doc_path)

    if rostro_img is None or doc_img is None:
        messagebox.showerror("Error", "No se pudieron cargar las imágenes.")
        return

    rostro_crop = detect_faces(rostro_img)
    doc_crop = detect_faces(doc_img)

    if rostro_crop is None or doc_crop is None:
        messagebox.showerror("Fallo en detección", "No se detectó rostro en una de las imágenes.")
        return

    match, score = compare_faces(rostro_crop, doc_crop)
    match_lm, score_lm = compare_faces_with_landmarks(rostro_img, doc_img)

    resultado = (
        f"[ArcFace] Similitud: {score:.4f} → {'MATCH' if match else 'NO MATCH'}\n"
        f"[Landmarks] Similitud: {score_lm:.4f} → {'MATCH' if match_lm else 'NO MATCH'}"
    )
    messagebox.showinfo("Resultado", resultado)

# GUI principal
app = tk.Tk()
camera_id_var = tk.IntVar(value=0)
app.title("Verificación Facial Local")
frame_cam = tk.LabelFrame(app, text="Selecciona cámara", padx=10, pady=10)
frame_cam.pack(padx=10, pady=5, fill="x")

tk.Label(frame_cam, text="ID de cámara:").pack(side="left")
tk.OptionMenu(frame_cam, camera_id_var, *range(3)).pack(side="left", padx=10)

app.geometry("400x350")

frame1 = tk.LabelFrame(app, text="Captura o carga de imágenes", padx=10, pady=10)
frame1.pack(padx=10, pady=10, fill="both")

# Botones rostro
tk.Button(frame1, text="📸 Capturar rostro", command=capturar_rostro).pack(fill="x", pady=2)
tk.Button(frame1, text="📂 Cargar imagen rostro", command=cargar_rostro).pack(fill="x", pady=2)
lbl_estado_rostro = tk.Label(frame1, text="Rostro: no cargado")
lbl_estado_rostro.pack(pady=2)

# Botones documento
tk.Button(frame1, text="📸 Capturar documento", command=capturar_documento).pack(fill="x", pady=2)
tk.Button(frame1, text="📂 Cargar imagen documento", command=cargar_documento).pack(fill="x", pady=2)
lbl_estado_doc = tk.Label(frame1, text="Documento: no cargado")
lbl_estado_doc.pack(pady=2)

# Comparación
tk.Button(app, text="🧠 Comparar imágenes", command=comparar, bg="#4CAF50", fg="white").pack(pady=10, fill="x", padx=20)

app.mainloop()
