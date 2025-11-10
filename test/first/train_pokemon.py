# train_pokemon.py (Simulação de treinamento)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pokemon_cnn import PokemonCNN # Importa o modelo

# --- Hiperparâmetros ---
BATCH_SIZE = 64
NUM_EPOCHS = 5 # Aumente para treinar de verdade
LEARNING_RATE = 0.001
SAVE_PATH = 'pokemon_cnn.pt'

# --- Configuração de Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preparação dos Dados ---
# ATENÇÃO: Substitua esta seção pela forma real como você carrega seu dataset de Pokémons.
# Se seu dataset estiver em uma pasta 'data/pokemons' com subpastas por classe:
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalização se necessário
])

# Simulação: vamos apenas criar um dataset dummy se não tiver o real
try:
    # Tente carregar o dataset real (exemplo com ImageFolder)
    train_data = datasets.ImageFolder(root='../DATASETS/all_pokemon', transform=transform)
except:
    # Se falhar, crie um dataset dummy para fins didáticos
    print("AVISO: Dataset real não encontrado. Criando dataset dummy...")
    train_data = datasets.FakeData(size=1000, image_size=(3, 64, 64), num_classes=1000, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# --- Inicialização ---
model = PokemonCNN(num_classes=len(train_data.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Treinamento (Simulado) ---
print(f"Iniciando treinamento com {len(train_data.classes)} classes...")
model.train()
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward e Otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# --- Salvar Modelo ---
torch.save(model.state_dict(), SAVE_PATH)
print(f"Modelo salvo em: {SAVE_PATH}")
