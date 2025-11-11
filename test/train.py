# ============================================================
# train.py — Treino do modelo Pokémon (ResNet34)
# Dataset: Roboflow Pokedex (somente train/ é usado)
# ============================================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet34_Weights
from dataset import load_pokemon_dataset

# ------------------------------------------------------------
# CONFIGURAÇÕES
# ------------------------------------------------------------

DATASET_PATH = "/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14/"
MODEL_SAVE_PATH = "pokemon_model_final.pth"
MODEL_BEST_SAVE_PATH = "pokemon_model_best.pth"
STATS_DIR = "debug_outputs"

NUM_EPOCHS = 40
BATCH_SIZE = 64
LR_HEAD = 5e-4         # treinar apenas a FC + layer4
LR_FINETUNE = 1e-5     # liberar rede inteira
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device detectado: {DEVICE}")

# ------------------------------------------------------------
# CARREGAR MEAN / STD + CLASSES
# ------------------------------------------------------------

stats_path = os.path.join(STATS_DIR, "dataset_stats.json")
classes_path = os.path.join(STATS_DIR, "classes.json")

if not os.path.exists(stats_path) or not os.path.exists(classes_path):
    print("[INFO] Calculando mean/std do dataset...")
    _, _, _, classes, mean, std = load_pokemon_dataset(
        base_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        compute_and_save_stats=True,
        stats_out_dir=STATS_DIR,
        auto_create_valid=True,          # <-- estratificação automática
    )
else:
    print("[INFO] Carregando mean/std existentes...")
    with open(stats_path) as f:
        s = json.load(f)
        mean, std = s["mean"], s["std"]
    with open(classes_path) as f:
        classes = json.load(f)["classes"]

NUM_CLASSES = len(classes)
print(f"[INFO] Total de classes detectadas: {NUM_CLASSES}")

# ------------------------------------------------------------
# TRANSFORMAÇÕES (remove bordas pretas do Roboflow FIT)
# ------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ------------------------------------------------------------
# CARREGAR DATASET (AGORA, SPLIT AUTOMÁTICO)
# ------------------------------------------------------------

train_loader, valid_loader, test_loader, classes, _, _ = load_pokemon_dataset(
    base_path=DATASET_PATH,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    train_transform=train_transform,
    test_transform=test_transform,
    auto_create_valid=True,          # <-- recria valid/test estratificado!
)

# ------------------------------------------------------------
# INICIALIZAR MODELO (ResNet34 pré-treinada)
# ------------------------------------------------------------

print("\n[INFO] Carregando ResNet34 pré-treinada...")
model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

# substitui a FC para o número de classes
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# congela tudo exceto layer4 e a FC
for name, param in model.named_parameters():
    param.requires_grad = ("layer4" in name) or ("fc" in name)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)

best_val_acc = 0.0
epochs_no_improve = 0
EARLY_STOPPING_PATIENCE = 8

# ------------------------------------------------------------
# TREINO (FASE 1: só FC + layer4)
# ------------------------------------------------------------

print("\n[INFO] Fase 1 — Treinando FC + layer4\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # validação
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            predicted = model(imgs).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    avg_loss = running_loss / len(train_loader)

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_BEST_SAVE_PATH)
        print(f"   ✅ Novo melhor modelo salvo ({best_val_acc:.4f})")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print("\n[INFO] Early stopping ativado!")
        break

# ------------------------------------------------------------
# TREINO (FASE 2: liberar toda a rede — fine-tuning total)
# ------------------------------------------------------------

print("\n[INFO] Fase 2 — Fine-tuning de toda a ResNet\n")

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE)

for epoch in range(10):  # fine-tune curto
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"[FineTune {epoch+1}/10] Loss: {running_loss / len(train_loader):.4f}")

# ------------------------------------------------------------
# SALVAR MODELO FINAL
# ------------------------------------------------------------

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("\n✅ Treino concluído!")
print(f"📌 Modelo final salvo em: {MODEL_SAVE_PATH}")
print(f"🏆 Melhor modelo (valid) salvo em: {MODEL_BEST_SAVE_PATH}")
print(f"🔥 Melhor Val Acc: {best_val_acc:.4f}")
