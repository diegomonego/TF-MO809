# dataset.py
import os
import json
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def compute_mean_std(loader):
    """Compute dataset mean and std (images in [0,1])"""
    import torch
    mean = torch.zeros(3)
    std = torch.zeros(3)
    count = 0
    for imgs, _ in loader:
        mean += imgs.mean([0,2,3])
        std += imgs.std([0,2,3])
        count += 1
    mean /= max(1, count)
    std /= max(1, count)
    return mean.tolist(), std.tolist()

def load_pokemon_dataset(
    base_path="/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14",
    batch_size=32,
    num_workers=0,
    img_size=224,
    train_transform=None,
    test_transform=None,
    compute_and_save_stats=True,
    stats_out_dir="debug_outputs",
    min_images_per_class_check=0,   # se >0 faz assert mínimo por classe
    auto_create_valid=False,        # se True, cria valid a partir de train se faltar
    valid_frac=0.15,
    seed=42
):
    """
    Retorna: train_loader, valid_loader, test_loader, classes
    Expectativa de estrutura:
      base_path/
         train/
           ClassA/
           ClassB/
         valid/
         test/
    Se valid/test não existirem e auto_create_valid=True, cria valid split a partir de train.
    """

    os.makedirs(stats_out_dir, exist_ok=True)

    train_dir = os.path.join(base_path, "train")
    valid_dir = os.path.join(base_path, "valid")
    test_dir  = os.path.join(base_path, "test")

    # checagens iniciais
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train directory não encontrado em: {train_dir}")

    # default transforms se não fornecidos
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.1),
            transforms.ToTensor(),   # -> [0,1]
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    # Carregar dataset train (ImageFolder)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)

    # Se valid não existe e auto_create_valid, cria indices
    if auto_create_valid:
        print("[DATASET] valid/ não encontrado — criando valid split a partir de train/")
        # criamos índices de validação por classe para manter balance
        class_to_indices = {}
        for idx, (_, label) in enumerate(train_dataset.imgs):
            class_to_indices.setdefault(label, []).append(idx)

        valid_indices = []
        train_indices = []
        random.seed(seed)
        for label, idxs in class_to_indices.items():
            n_valid = max(1, int(len(idxs) * valid_frac))
            random.shuffle(idxs)
            valid_indices.extend(idxs[:n_valid])
            train_indices.extend(idxs[n_valid:])

        # Subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        valid_dataset = Subset(datasets.ImageFolder(root=train_dir, transform=test_transform), valid_indices)
        # note: test remains None unless exists on disk
        test_dataset = None
    else:
        # se valid existe, carregue normalmente
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=test_transform) if os.path.isdir(valid_dir) else None
        test_dataset  = datasets.ImageFolder(root=test_dir,  transform=test_transform) if os.path.isdir(test_dir) else None

    # Criar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if valid_dataset is not None else None
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_dataset is not None else None

    # recuperar classes: se train_dataset for Subset, pegamos .dataset.classes
    if hasattr(train_dataset, "dataset"):
        classes = train_dataset.dataset.classes
    else:
        classes = train_dataset.classes

    # opcional: checar min images por classe
    if min_images_per_class_check > 0:
        # contar arquivos por classe na pasta train_dir
        class_counts = {}
        for cls in classes:
            p = os.path.join(train_dir, cls)
            class_counts[cls] = len([name for name in os.listdir(p) if os.path.isfile(os.path.join(p, name))]) if os.path.isdir(p) else 0
        too_small = {c:n for c,n in class_counts.items() if n < min_images_per_class_check}
        if too_small:
            print("[DATASET] Classes com poucas imagens (< min):", too_small)
        else:
            print("[DATASET] Todas as classes têm >= min_images_per_class_check imagens.")

    # Calcular mean/std se requerido (usando test_transform sem normalização)
    if compute_and_save_stats:
        # cria um loader temporário sem augment com ToTensor()
        from torchvision import transforms as T
        tmp_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        tmp_loader = DataLoader(datasets.ImageFolder(root=train_dir, transform=tmp_transform), batch_size=64, shuffle=False, num_workers=0)
        mean, std = compute_mean_std(tmp_loader)
        stats_path = os.path.join(stats_out_dir, "dataset_stats.json")
        with open(stats_path, "w") as f:
            json.dump({"mean": mean, "std": std}, f)
        classes_path = os.path.join(stats_out_dir, "classes.json")
        with open(classes_path, "w") as f:
            json.dump({"classes": classes}, f)
        print(f"[DATASET] stats salvos em: {stats_path}")
        print(f"[DATASET] classes salvos em: {classes_path}")
    else:
        mean = std = None

    print(f"[DATASET] Pokémon carregado com sucesso — {len(classes)} classes.")
    # tamanho dos datasets (tratando Subset possivelmente)
    def _len(ds):
        try:
            return len(ds)
        except Exception:
            return "unknown"
    print(f"Train size: {_len(train_dataset)}")
    print(f"Valid size: {_len(valid_dataset)}")
    print(f"Test size:  {_len(test_dataset)}")

    return train_loader, valid_loader, test_loader, classes, mean, std
