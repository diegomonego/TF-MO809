import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet34_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import json

from dataset import load_pokemon_dataset   # ← usa a mesma função do train.py


def plot_confusion_matrix(cm, classes, out_path, normalize=False):
    plt.figure(figsize=(14, 14))

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=4)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    import argparse
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default="confusion.png")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Carregando modelo: {args.model}")

    # load stats and classes
    stats = json.load(open("debug_outputs/dataset_stats.json"))
    mean, std = stats["mean"], stats["std"]

    classes = json.load(open("debug_outputs/classes.json"))["classes"]
    NUM_CLASSES = len(classes)

    DATASET_PATH = "/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14"

    print("[INFO] Carregando dataset...")

    train_loader, valid_loader, test_loader, classes, mean, std = load_pokemon_dataset(
        base_path=DATASET_PATH,
        batch_size=args.batch,
        img_size=224,
        auto_create_valid=True,     # mantém a mesma divisão do treino
        compute_and_save_stats=False
    )

    # como não existe test/, usamos a valid como test
    test_loader = valid_loader

    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("[INFO] Avaliando no test set...")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1

    plot_confusion_matrix(cm, classes, args.out, normalize=True)

    print(f"\n✅ Confusion matrix salva em: {args.out}")
