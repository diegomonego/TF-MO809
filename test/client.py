# client.py (Adaptado para encontrar amostra de baixa perda)
import os
import json
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- IMPORTADO ADICIONALMENTE
from torchvision import transforms, models
from torchvision.models import resnet34, ResNet34_Weights
from dataset import load_pokemon_dataset
from utils import send_data

# --- Configurações ---
HOST = '127.0.0.1'
PORT = 65432
MODEL_PATH = 'pokemon_model_best.pth'
DATASET_PATH = '/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14'
IMG_SIZE = 224
# SECRET_IMAGE_IDX = 42  <--- REMOVIDO! Será encontrado dinamicamente.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CLIENT] Usando device: {device}")

# --- Carregar mean/std ---
stats_path = os.path.join("debug_outputs", "dataset_stats.json")
if not os.path.exists(stats_path):
    raise FileNotFoundError(f"{stats_path} não encontrado. Rode train.py antes.")
with open(stats_path, "r") as f:
    stats = json.load(f)
mean, std = stats["mean"], stats["std"]
print(f"[CLIENT] Usando mean={mean} std={std}")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --- Carregar dataset ---
train_loader, valid_loader, test_loader, classes, _, _ = load_pokemon_dataset(
    base_path=DATASET_PATH,
    batch_size=1,
    num_workers=0,
    train_transform=transform,
    test_transform=transform,
    auto_create_valid=True,   # ✅ somente isso
)

if valid_loader is None:
    raise RuntimeError(
        "[CLIENT] ERRO: valid_loader não foi criado! Verifique se o dataset tem /valid ou use auto_create_valid=True."
    )

# --- Carregar modelo ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
num_classes = len(classes)
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("[CLIENT] Modelo carregado com sucesso.")


# ===================================================================
# --- (NOVA LÓGICA) Encontrar imagem de baixa perda na VALIDAÇÃO ---
# ===================================================================
print("[CLIENT] Procurando uma imagem de 'baixa perda' (classificada corretamente) na VALIDAÇÃO...")

dataset_obj = valid_loader.dataset  # Usar o dataset de validação
img_true = None
label_true_idx = -1
pokemon_name = ""
SECRET_IMAGE_IDX = -1  # Este será o índice *relativo* ao valid_dataset
CONFIDENCE_THRESHOLD = 0.98  # Limiar de confiança

for idx, (img, label) in enumerate(dataset_obj):
    img_batch = img.unsqueeze(0).to(device)
    label_tensor = torch.tensor([label], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(img_batch)
        pred_label = output.argmax(dim=1)
        confidence = F.softmax(output, dim=1).max().item()

    # CONDIÇÃO: O modelo acertou E está confiante?
    if pred_label == label_tensor and confidence > CONFIDENCE_THRESHOLD:
        SECRET_IMAGE_IDX = idx  # Salva o índice relativo
        img_true = img_batch      # Salva o tensor (já no device e com batch dim)
        label_true_idx = label    # Salva o rótulo (int)
        pokemon_name = classes[label]
        break  # Para no primeiro que encontrar

if SECRET_IMAGE_IDX == -1:
    raise RuntimeError(f"Não foi possível encontrar uma imagem de baixa perda! Tente baixar o CONFIDENCE_THRESHOLD (ex: 0.9).")

print(f"[CLIENT] Imagem de baixa perda encontrada! Usando (índice relativo da validação): idx={SECRET_IMAGE_IDX}, label={label_true_idx} ({pokemon_name})")
# As variáveis img_true, label_true_idx, pokemon_name, e SECRET_IMAGE_IDX
# agora estão definidas para o resto do script.
# ===================================================================
# --- (FIM DA NOVA LÓGICA) ---
# ===================================================================


# --- Calcular gradientes (mantém seu código) ---
criterion = nn.CrossEntropyLoss()
model.zero_grad()
output = model(img_true) # Já foi calculado, mas calculamos de novo para obter gradientes
label_tensor = torch.tensor([label_true_idx], dtype=torch.long).to(device)
loss = criterion(output, label_tensor)
loss.backward()

# --- Construir lista robusta de gradientes + metadados ---
target_grads = []
param_shapes = []
param_names = []

# percorre parâmetros na mesma ordem que model.parameters()
for name, p in model.named_parameters():
    param_names.append(name)
    param_shapes.append(tuple(p.data.shape))
    if p.grad is None:
        # cria um array zeros com shape igual ao parâmetro
        zeros = np.zeros(tuple(p.data.shape), dtype=np.float32)
        target_grads.append(zeros)
    else:
        g = p.grad.detach().cpu().numpy().astype('float32')
        # garantir contiguidade e tipo
        g = np.ascontiguousarray(g)
        target_grads.append(g)

print("[CLIENT] Gradientes calculados e convertidos.")
print(f"[CLIENT] Número de parâmetros: {len(param_names)}")

# --- Recuperar caminho absoluto da imagem secreta (para enviar ao servidor) ---
secret_image_path = None
try:
    ds = dataset_obj
    # Subset?
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        idx_in_base = ds.indices[SECRET_IMAGE_IDX]
    else:
        base = ds
        idx_in_base = SECRET_IMAGE_IDX

    # ImageFolder-style: samples or imgs attribute
    if hasattr(base, "samples") and len(base.samples) > idx_in_base:
        secret_image_path = os.path.abspath(base.samples[idx_in_base][0])
    elif hasattr(base, "imgs") and len(base.imgs) > idx_in_base:
        secret_image_path = os.path.abspath(base.imgs[idx_in_base][0])
    else:
        secret_image_path = None
except Exception as e:
    print("[CLIENT] Warning: não foi possível recuperar caminho da imagem:", e)
    secret_image_path = None

# --- Montar payload robusto ---
payload = {
    "model_arch": "resnet34",
    "gradients": target_grads,        # lista de numpy arrays float32
    "param_names": param_names,       # lista de strings
    "param_shapes": param_shapes,     # lista de tuples (para checagem no servidor)
    "label_idx": int(label_true_idx),
    "label_name": str(pokemon_name),
    "secret_image_idx": int(SECRET_IMAGE_IDX),
    "secret_image_path": secret_image_path,
    "dataset_stats_path": os.path.abspath(stats_path)
}

# --- Enviar ---
print(f"[CLIENT] Conectando ao servidor {HOST}:{PORT} ...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    send_data(s, payload)
    print("[CLIENT] Dados enviados com sucesso.")
