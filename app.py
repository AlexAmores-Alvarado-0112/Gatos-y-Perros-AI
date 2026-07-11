from flask import Flask, render_template, request, send_from_directory
from predict import predecir_imagen

import os
import random
import time

app = Flask(__name__)

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

    clase, confianza = predecir_imagen(ruta)

    tiempo = round(time.time() - inicio,3)

    # -------------------------------
    # Seleccionar imagen representativa
    # -------------------------------

    imagen_modelo = "img/desconocido/desconocido.png"

    if clase == "Gatos":

        carpeta = os.path.join(BASE_DIR,"static","img","gatos")

    elif clase == "perros":

        carpeta = os.path.join(BASE_DIR,"static","img","perros")

    else:

        carpeta = os.path.join(BASE_DIR,"static","img","desconocido")

    if os.path.exists(carpeta):

        archivos = [

            f for f in os.listdir(carpeta)

            if f.lower().endswith((".jpg",".jpeg",".png"))

        ]

        if len(archivos) > 0:

            nombre = random.choice(archivos)

            imagen_modelo = os.path.join(

                "img",

                os.path.basename(carpeta),

                nombre

            ).replace("\\","/")

    print("Clase:",clase)

    print("Imagen:",imagen_modelo)

    return render_template(

        "index.html",

        clase=clase,

        confianza=confianza,

        tiempo=tiempo,

        imagen=archivo.filename,

        imagen_modelo=imagen_modelo

    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )


if __name__ == "__main__":

    app.run(debug=True)