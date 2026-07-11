import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

print("Paso 1: Iniciando...")

# ==================================================
# Configuración
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASES = ["Gatos", "perros"]

# Umbral mínimo de confianza
UMBRAL_CONFIANZA = 90

# ==================================================
# Arquitectura del modelo
# ==================================================

class MiCNNClasificador(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,2)

        )

    def forward(self,x):

        x = self.features(x)
        x = x.view(x.size(0),-1)

        return self.classifier(x)

# ==================================================
# Cargar modelo
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR,"modelo_animales.pth")

print("Buscando modelo en:")
print(RUTA_MODELO)

model = MiCNNClasificador().to(device)

model.load_state_dict(
    torch.load(RUTA_MODELO,map_location=device)
)

model.eval()

print("Modelo cargado correctamente.")

# ==================================================
# Transformaciones
# ==================================================

transformacion = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ==================================================
# Predicción
# ==================================================

def predecir_imagen(ruta_imagen):

    imagen = Image.open(ruta_imagen).convert("RGB")

    imagen = transformacion(imagen)

    imagen = imagen.unsqueeze(0).to(device)

    with torch.no_grad():

        salida = model(imagen)

        probabilidades = torch.softmax(salida, dim=1)

        confianza, indice = torch.max(probabilidades,1)

    confianza = round(confianza.item()*100,2)

    # Si la confianza es baja
    if confianza < UMBRAL_CONFIANZA:

        return "Desconocido", confianza

    clase = CLASES[indice.item()]

    return clase, confianza