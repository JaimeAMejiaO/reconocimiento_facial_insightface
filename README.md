# 🔐 Verificación de Identidad Mediante Inteligencia Artificial

Este proyecto implementa un sistema **local** de verificación facial para validar la identidad de una persona comparando su rostro con la fotografía contenida en un documento de identidad (Cualquier documento legal). El sistema fue desarrollado como parte de un proyecto académico de maestría, con proyección futura para su uso en campo real.

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
| **NVIDIA CUDA** | Toolkit para uso de GPU |
| **Microsoft Visual C++ Build Tools ** | Ciertas dependencias requieren compilación de módulos en C++ | 

---

💻 Requisitos de hardware y aceleración por GPU
Este sistema fue diseñado para ejecutarse localmente aprovechando la aceleración por GPU, lo cual es altamente recomendado (aunque también puede ejecutarse con CPU a menor velocidad).

⚠️ Para ejecutar el sistema con GPU (CUDA), es necesario tener:

Una tarjeta gráfica NVIDIA compatible

CUDA Toolkit (versión 12.x o superior)

cuDNN correspondiente a la versión CUDA instalada

Drivers NVIDIA actualizados

La inferencia con onnxruntime y insightface aprovechará automáticamente la GPU si está correctamente configurada.
Sin estos componentes, el sistema caerá automáticamente al modo CPUExecutionProvider, funcionando más lento.

---

🔧 Microsoft Visual C++ Build Tools

Algunas dependencias como insightface requieren compilación de módulos en C++.

Para ello debes instalar:

👉 Microsoft C++ Build Tools

Durante la instalación, selecciona al menos:

C++ build tools

Windows 10 SDK (o superior)

MSVC v14.x

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
