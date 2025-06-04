# Modelos usados en la Verificación Facial

Esta carpeta contiene (o debe contener) los modelos necesarios para ejecutar el sistema de verificación facial local basado en ArcFace y RetinaFace.

---

## 📘 ArcFace – `arcface_resnet100.onnx`

- **Función**: Extracción de embeddings para comparación facial.
- **Formato**: ONNX
- **Tamaño aproximado**: 255 MB
- **Descarga directa**: [Google Drive - ArcFace ResNet100](https://drive.google.com/...) *(agrega el enlace cuando lo subas)*

🔹 **Ubicación esperada:**


---

## 📘 RetinaFace – `retinaface_resnet50.pth`

- **Función**: Detección de rostro y landmarks para alineamiento.
- **Formato**: PyTorch `.pth`
- **Tamaño aproximado**: 107 MB
- **Descarga directa**: [Google Drive - RetinaFace ResNet50](https://drive.google.com/...) *(agrega el enlace cuando lo subas)*

🔹 **Ubicación esperada:**


---

## 📦 Otros modelos descargados automáticamente (buffalo_l)

El subdirectorio:


Es generado automáticamente cuando InsightFace descarga modelos preentrenados por defecto, como respaldo o referencia. Aunque en esta versión estamos usando `arcface_resnet100.onnx`, puedes mantener esta carpeta por compatibilidad.

---

## ❗ Importante

Estos archivos **no están incluidos en el repositorio de GitHub** por restricciones de tamaño (límite de 100 MB por archivo). Si clonas este proyecto, debes:

1. Descargar los archivos desde los enlaces indicados.
2. Colocarlos en la ruta especificada dentro de la carpeta `models/`.


---

📌 **No elimines esta carpeta `models/`** aunque esté vacía en el repositorio, ya que es esencial para la carga correcta de modelos en tiempo de ejecución.
