#!/usr/bin/env python3
import os
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights
import torch.nn.functional as F
import argparse
import json



def load_classes(classes_path="debug_outputs/classes.json"):
    with open(classes_path, "r") as f:
        return json.load(f)["classes"]




def build_transform(mean, std, img_size=224):
    return transforms.Compose([
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    ])




def predict(image_path, model_path, classes, mean, std, device):
    transform = build_transform(mean, std)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)


    # carregar modelo
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()


    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)


    top5_prob, top5_idx = torch.topk(probs, 5)
    results = [(classes[i], float(top5_prob[j])) for j, i in enumerate(top5_idx)]
    return results




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path para o state_dict do modelo (.pth)")
    parser.add_argument("--img", required=True, help="Caminho para a imagem a ser classificada")
    parser.add_argument("--classes", default="debug_outputs/classes.json", help="JSON com classes")
    parser.add_argument("--stats", default="debug_outputs/dataset_stats.json", help="JSON com mean/std")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    classes = load_classes(args.classes)
    with open(args.stats, "r") as f:
        s = json.load(f)
        mean, std = s["mean"], s["std"]


    top5 = predict(args.img, args.model, classes, mean, std, device)
    print("Top-5 predictions (class, prob):")
    for cls, p in top5:
        print(f" {cls}: {p:.4f}")