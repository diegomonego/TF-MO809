# train.py (Versão 2 - Lendo de CSV)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image # Para carregar imagens
import pandas as pd # Para ler o CSV
import os
from model import PokemonCNN # Importa nossa classe de modelo

# --- Configurações ---
# ATENÇÃO: Ajuste estes caminhos!
CSV_PATH = '../DATASETS/Pokemon.csv'    # Caminho para o seu pokemon.csv
IMAGE_DIR = '../DATASETS/all_pokemon/'  # Caminho para a pasta com TODAS as imagens
MODEL_SAVE_PATH = 'pokemon_model.pth'

# Hiperparâmetros de Treino
NUM_EPOCHS = 30      # Aumentado para melhor performance
BATCH_SIZE = 64      # Aumentado para aproveitar a GPU (pode ser 128 se a VRAM aguentar)
LEARNING_RATE = 0.001
IMG_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

# --- 1. Classe de Dataset Customizada ---
# Esta classe vai "ensinar" o PyTorch a ler seu CSV
class PokemonCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # O ImageFolder cria um "índice de classes" (ex: 'Pikachu' -> 0)
        # Vamos fazer o mesmo.
        self.classes = self.data_frame['name'].unique() # Pega todos os nomes únicos
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Dataset encontrado. Total de {len(self.data_frame)} imagens.")
        print(f"Total de {len(self.classes)} classes únicas.")

    def __len__(self):
        # Retorna o número total de imagens no CSV
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Pega uma linha do CSV (ex: a linha 'idx')
        img_row = self.data_frame.iloc[idx]
        
        # --- ATENÇÃO: Verifique as colunas do seu CSV! ---
        # 1. Pegue o NOME do Pokémon (o rótulo)
        pokemon_name = img_row['name'] # Estou assumindo que a coluna é 'name'
        label = self.class_to_idx[pokemon_name] # Converte nome para índice (ex: 25)
        
        # 2. Monte o NOME DO ARQUIVO da imagem
        # Estou assumindo que o CSV tem uma coluna com o número (ex: '#')
        # e que os arquivos são 'NUMERO.jpg'
        try:
            # Tenta usar a coluna '#' (Pokedex number)
            img_filename = str(img_row['number']) + '.jpg'
        except KeyError:
            # Se não houver '#', tenta outra coluna (ex: 'filename')
            # img_filename = img_row['filename'] # Ajuste conforme necessário
            print("ERRO: Não encontrei a coluna 'number' no CSV para o nome do arquivo.")
            return None, None
            
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            # Carrega a imagem
            image = Image.open(img_path).convert('RGB') # Garante 3 canais (RGB)
        except FileNotFoundError:
            print(f"AVISO: Imagem não encontrada em {img_path}. Pulando.")
            # Retorna a próxima imagem para evitar erro
            return self.__getitem__((idx + 1) % len(self)) 

        # Aplica as transformações (Resize, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. Preparação dos Dados ---

# Data Augmentation: Isso ajuda o modelo a não "decorar"
# Vamos adicionar flips e rotações aleatórias
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(), # Vira a imagem horizontalmente
    T.RandomRotation(10),     # Gira em até 10 graus
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

try:
    train_dataset = PokemonCSVDataset(csv_file=CSV_PATH, 
                                      img_dir=IMAGE_DIR, 
                                      transform=transform)
    
    # Pega o número de classes que o dataset encontrou
    NUM_CLASSES = len(train_dataset.classes) 
    print(f"Número de classes ajustado para: {NUM_CLASSES}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
except FileNotFoundError:
    print(f"ERRO: Arquivo CSV não encontrado em {CSV_PATH}")
    exit()
except Exception as e:
    print(f"ERRO ao carregar o dataset: {e}")
    print("Verifique os nomes das colunas ('name', 'number') no script.")
    exit()


# --- 3. Treinamento ---
model = PokemonCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Iniciando treinamento (isso pode demorar mais agora, o que é bom)...")
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # 'labels' pode ser None se uma imagem falhou ao carregar
        if images is None or labels is None:
            continue
            
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 50 == 0: # Reporta a cada 50 batches
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f'Fim da Epoch [{epoch+1}/{NUM_EPOCHS}], Loss Média: {total_loss/len(train_loader):.4f}')

# --- 4. Salvar Modelo ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nTreinamento concluído. Modelo robusto salvo em {MODEL_SAVE_PATH}")