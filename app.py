from flask import Flask, render_template, request, send_from_directory
from predict import predecir_imagen

import os
import random
import time
import gc  # <--- Forzar la liberación de memoria RAM

app = Flask(__name__)

# ==================================================
# Configuración estricta para Producción (Baja RAM)
# ==================================================
app.config["DEBUG"] = False
app.debug = False

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "imagen" not in request.files:
        return "No se encontró ninguna imagen."

    archivo = request.files["imagen"]

    if archivo.filename == "":
        return "No seleccionaste ninguna imagen."

    ruta = os.path.join(app.config["UPLOAD_FOLDER"], archivo.filename)
    archivo.save(ruta)

    inicio = time.time()

    # Ejecuta la predicción del modelo
    clase, confianza = predecir_imagen(ruta)

    tiempo = round(time.time() - inicio, 3)

    # -------------------------------
    # Seleccionar imagen representativa
    # -------------------------------
    imagen_modelo = "img/desconocido/desconocido.png"

    # Convertimos a minúsculas para evitar problemas de mayúsculas/minúsculas
    clase_normalizada = clase.lower()

    if clase_normalizada == "gatos":
        carpeta = os.path.join(BASE_DIR, "static", "img", "gatos")
    elif clase_normalizada == "perros":
        carpeta = os.path.join(BASE_DIR, "static", "img", "perros")
    else:
        carpeta = os.path.join(BASE_DIR, "static", "img", "desconocido")

    if os.path.exists(carpeta):
        archivos = [
            f for f in os.listdir(carpeta)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(archivos) > 0:
            nombre = random.choice(archivos)
            imagen_modelo = os.path.join(
                "img",
                os.path.basename(carpeta),
                nombre
            ).replace("\\", "/")

    print("Clase:", clase)
    print("Imagen:", imagen_modelo)

    # 🛠️ Limpieza post-predicción: Borramos variables pesadas de la petición
    # y forzamos al recolector de basura a limpiar la RAM de inmediato.
    del archivo
    gc.collect()

    return render_template(
        "index.html",
        clase=clase,
        confianza=confianza,
        tiempo=tiempo,
        imagen=archivo.filename if 'archivo' in locals() else "",
        imagen_modelo=imagen_modelo
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


if __name__ == "__main__":
    # Si se ejecuta de forma local o directa
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False  
    )