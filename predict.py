import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

print("Paso 1: Iniciando Predictor...")

# ==================================================
# Configuración
# ==================================================
device = torch.device("cpu")

# IMPORTANTE: Asegúrate de que el orden de esta lista coincida exactamente con
# el orden alfabético de tus carpetas en Google Drive (ej: Gatos, Otros, perros)
CLASES = ["Gatos", "Otros", "perros"]

# Umbral mínimo de confianza para evitar falsos positivos
UMBRAL_CONFIANZA = 80.0

# ==================================================
# Arquitectura del modelo (Modificada para 3 clases)
# ==================================================
class MiCNNClasificadorMejorada(nn.Module):
    def __init__(self):
        super(MiCNNClasificadorMejorada, self).__init__()

        # Extractor de características más profundo
        self.features = nn.Sequential(
            # Bloque 1: 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 2: 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 3: 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Bloque de Clasificación
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 3)   # <--- CAMBIADO A 3 CLASES
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================================================
# Cargar modelo (Optimizado para baja RAM)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_animales.pth")

print("Buscando modelo en:")
print(RUTA_MODELO)

model = MiCNNClasificadorMejorada().to(device)

model.load_state_dict(
    torch.load(RUTA_MODELO, map_location=device, weights_only=True)
)
model.eval()

# Desactivamos el cálculo de gradientes de forma global
for param in model.parameters():
    param.requires_grad = False

print("Modelo de 3 clases cargado correctamente.")

# ==================================================
# Transformaciones de producción
# ==================================================
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
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
        confianza, indice = torch.max(probabilidades, 1)

    confianza = round(confianza.item() * 100, 2)
    clase = CLASES[indice.item()]

    # Si la predicción cae en la carpeta de "Otros" o la confianza es inferior al umbral,
    # forzamos el retorno a "Desconocido".
    if clase.lower() == "otros" or confianza < UMBRAL_CONFIANZA:
        return "Desconocido", confianza

    return clase, confianza