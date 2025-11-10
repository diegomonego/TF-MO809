# client.py (Versão 2 - Lendo de CSV)
import torch
import torch.nn as nn
import torchvision.transforms as T
import socket
from model import PokemonCNN
from utils import send_data
from dataset import PokemonCSVDataset # <-- IMPORTA NOSSA NOVA CLASSE

# --- Configurações ---
HOST = '127.0.0.1'  # IP do Servidor
PORT = 65432        # Porta do Servidor
MODEL_PATH = 'pokemon_model.pth'
IMG_SIZE = 64
SECRET_IMAGE_IDX = 42 # O índice da imagem que queremos "vazar"

# --- ATENÇÃO: Ajuste os caminhos para o NOVO dataset ---
CSV_PATH = '../DATASETS/Pokemon.csv'    # Caminho para o seu pokemon.csv
IMAGE_DIR = '../DATASETS/all_pokemon/'  # Caminho para a pasta com TODAS as imagens

device = torch.device("cpu") # Cliente não precisa de GPU
print(f"Cliente usando device: {device}")

# --- Carregar a Imagem Secreta (usando o novo Dataset) ---
# O transform do cliente NÃO deve ter Data Augmentation (RandomFlip, etc.)
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

try:
    # 1. Carrega o dataset (mesmo que só para pegar 1 imagem)
    secret_dataset = PokemonCSVDataset(csv_file=CSV_PATH, 
                                        img_dir=IMAGE_DIR, 
                                        transform=transform)
    
    # 2. Pega o NÚMERO DE CLASSES correto a partir do dataset
    NUM_CLASSES = len(secret_dataset.classes)
    
    # 3. Carrega a imagem e o rótulo
    img_true, label_true_idx = secret_dataset[SECRET_IMAGE_IDX]
    img_true = img_true.unsqueeze(0).to(device) # Adiciona dimensão de batch
    label_true = torch.tensor([label_true_idx], dtype=torch.long).to(device)
    
    pokemon_name = secret_dataset.idx_to_class[label_true_idx]
    print(f"Imagem secreta carregada (Índice {SECRET_IMAGE_IDX}, Classe: {label_true_idx} - {pokemon_name}).")

except Exception as e:
    print(f"ERRO ao carregar imagem secreta: {e}")
    print("Verifique os caminhos CSV_PATH e IMAGE_DIR.")
    exit()

# --- Carregar Modelo ---
# Carregamos o modelo DEPOIS de saber o NUM_CLASSES correto
model = PokemonCNN(num_classes=NUM_CLASSES).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Coloca o modelo em modo de avaliação
    criterion = nn.CrossEntropyLoss()
    print("Modelo treinado (pokemon_model.pth) carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Modelo treinado '{MODEL_PATH}' não encontrado.")
    print("Você precisa rodar o 'train.py' (versão 2) primeiro!")
    exit()
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    print("Isso pode acontecer se o NUM_CLASSES do treino for diferente do atual.")
    exit()


# --- Calcular Gradiente Secreto ---
model.zero_grad()
output = model(img_true)
loss = criterion(output, label_true)
loss.backward()

target_grads = [p.grad.detach().clone() for p in model.parameters()]
print("Gradiente secreto calculado.")

# --- Enviar para o Servidor ---
data_to_send = {
    'gradients': target_grads,
    'label_idx': label_true_idx,
    'label_name': pokemon_name, # Bônus: enviar o nome para o servidor
    'secret_image_idx': SECRET_IMAGE_IDX
}

print(f"Conectando ao servidor em {HOST}:{PORT}...")
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Conectado! Enviando gradientes...")
        send_data(s, data_to_send)
        print("Gradientes enviados. Fechando cliente.")

except ConnectionRefusedError:
    print(f"ERRO: Não foi possível conectar ao servidor. O 'server.py' está rodando?")
except Exception as e:
    print(f"ERRO de socket: {e}")