import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

print("Paso 1: Iniciando...")

# ==================================================
# Configuración
# ==================================================

# Forzar directamente CPU para ahorrar la memoria de inicialización de CUDA en Render
device = torch.device("cpu")

CLASES = ["Gatos", "perros"]

# Umbral mínimo de confianza
UMBRAL_CONFIANZA = 95

# ==================================================
# Arquitectura del modelo
# ==================================================

class MiCNNClasificadorMejorada(nn.Module):
    def __init__(self):
        super(MiCNNClasificadorMejorada, self).__init__()

        # Extractor de características más profundo
        self.features = nn.Sequential(
            # Bloque 1: 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # Estabiliza el entrenamiento
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 2: 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 3 nuevo: 56x56 -> 28x28 (Extrae características de más alto nivel)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Global Average Pooling: Reduce cualquier tamaño de mapa a 1x1 por canal
        # Esto evita las 100,000 neuronas y previene drásticamente el Overfitting
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Bloque de Clasificación mucho más ligero y eficiente
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), # Recibe solo los 64 canales del mapa final
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 2)   # Deja 2 si sigues con Perro/Gato, o 3 si aplicas la Opción 1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)       # Pasa por el Global Average Pool
        x = x.view(x.size(0), -1) # Ahora el aplanado es diminuto (solo 64 elementos)
        x = self.classifier(x)
        return x

# Instanciar el nuevo modelo
model = MiCNNClasificadorMejorada().to(device)
# ==================================================
# Cargar modelo (Optimizado para baja RAM)
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_animales.pth")

print("Buscando modelo en:")
print(RUTA_MODELO)

model = MiCNNClasificador().to(device)

# weights_only=True evita que PyTorch consuma RAM de más cargando metadatos innecesarios
model.load_state_dict(
    torch.load(RUTA_MODELO, map_location=device, weights_only=True)
)

model.eval()

# Desactivamos el cálculo de gradientes de forma global para el modelo cargado
for param in model.parameters():
    param.requires_grad = False

print("Modelo cargado correctamente.")

# ==================================================
# Transformaciones
# ==================================================

transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),           # Rota la imagen un poco
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Varía el color/brillo
    transforms.RandomHorizontalFlip(),        
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

    # Si la confianza es baja
    if confianza < UMBRAL_CONFIANZA:
        return "Desconocido", confianza

    clase = CLASES[indice.item()]

    return clase, confianza