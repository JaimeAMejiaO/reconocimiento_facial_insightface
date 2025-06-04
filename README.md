# 🔐 Verificación de Identidad Mediante Inteligencia Artificial

Este proyecto implementa un sistema **local** de verificación facial para validar la identidad de una persona comparando su rostro con la fotografía contenida en un documento de identidad (como una cédula o licencia). El sistema fue desarrollado como parte de un proyecto académico de maestría en Inteligencia Artificial, con proyección futura para su uso en campo real.

---

## 📷 ¿Cómo funciona?

1. El usuario captura o carga dos imágenes:
   - **Una imagen de su rostro**
   - **Una imagen del documento de identidad**

2. Se realiza:
   - **Detección facial** con `RetinaFace`
   - **Extracción de embeddings faciales** con `ArcFace` (ResNet-100)
   - **Comparación por similitud de coseno**
   - **Verificación del enfoque (no borroso)**
   - **Registro del resultado en CSV**

---

## 🧠 Tecnologías utilizadas

| Componente | Descripción |
|-----------|-------------|
| **RetinaFace** | Detector facial basado en redes convolucionales con landmarks 2D |
| **ArcFace ResNet100 (ONNX)** | Modelo de reconocimiento facial que usa margen angular aditivo |
| **OpenCV** | Captura desde cámara, procesamiento de imágenes y validación de nitidez |
| **Tkinter** | Interfaz gráfica local para pruebas |
| **ONNX Runtime** | Motor de inferencia para el modelo ArcFace |
| **Python** | Lenguaje de implementación principal |

---

## 🛠️ Estructura del proyecto

```bash
verificacion-facial-ia/
│
├── models/                     # Contiene arcface_resnet100.onnx y retinaface_resnet50.pth
│   └── (NO SE INCLUYEN en GitHub, descargar aparte)
│
├── modules/
│   ├── capture.py              # Captura y carga de imágenes
│   ├── detector.py             # Detección facial con RetinaFace
│   ├── recognizer.py           # Comparación de rostros con ArcFace
│   ├── preprocess.py           # Validaciones previas a comparación
│   ├── utils.py                # Validación de enfoque y logs
│   └── interface.py            # Interfaz gráfica
│
├── output/logs/               # CSVs de resultados con timestamp, score y match
├── data/input/                # Imágenes capturadas/cargadas localmente
├── .gitignore                 # Excluye modelos y entorno virtual
├── requirements.txt           # Librerías necesarias
└── README.md                  # Este archivo
