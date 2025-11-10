# client.py
import torch
import torch.nn as nn
import socket
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as T
from pokemon_cnn import PokemonCNN
from utils import send_data # Importa utilitário de envio

# --- Configurações de Comunicação ---
HOST = '127.0.0.1'  # Endereço IP do servidor
PORT = 65432        # Porta do servidor

# --- Setup do Modelo (Apenas para forward/backward) ---
device = torch.device("cpu") # Cliente não precisa de GPU

# O cliente carrega a estrutura do modelo, mas não precisa de pesos
model = PokemonCNN(num_classes=1000).to(device)
criterion = nn.CrossEntropyLoss()

# --- Seleção da Imagem Confidencial (Pokémon) ---
# Simulação: Gerar uma imagem dummy e um label
transform = T.Compose([T.ToTensor()])
# Imagem dummy 64x64, RGB
img_true = torch.rand(1, 3, 64, 64).to(device)
label_true = torch.tensor([42], dtype=torch.long).to(device) # Pokémon de índice 42

print(f"Imagem confidencial de 64x64, Label: {label_true.item()}")

# 1. Obter pesos iniciais/atuais (Simulação: vamos usar o checkpoint do treino)
try:
    model.load_state_dict(torch.load('pokemon_cnn.pt', map_location=device))
    print("Pesos do modelo carregados com sucesso do checkpoint.")
except:
    print("AVISO: Modelo não treinado (pokemon_cnn.pt não encontrado). Usando pesos aleatórios.")

# 2. Calcular o Gradiente Secreto
model.zero_grad()
model.eval() # Garantir que o modelo esteja em modo de avaliação (se não for Treinamento Federado)
loss = criterion(model(img_true), label_true)
loss.backward()

# Coletar os gradientes calculados (estes são os "dados vazados")
target_grads = [p.grad.detach().clone() for p in model.parameters()]
print(f"Gradientes calculados. Total de {len(target_grads)} tensores a enviar.")

# 3. Enviar o Gradiente ao Servidor
print(f"Conectando ao servidor em {HOST}:{PORT}...")
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Conexão estabelecida. Enviando gradientes...")
        send_data(s, target_grads)
        print("Gradientes enviados. Aguardando resultado...")
        # Você pode implementar uma lógica aqui para o servidor retornar a imagem reconstruída,
        # mas para este exemplo, vamos apenas fechar a conexão.
        print("Fim da comunicação com o cliente.")

except ConnectionRefusedError:
    print(f"ERRO: Não foi possível conectar ao servidor em {HOST}:{PORT}. Certifique-se que o server.py está rodando.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")
