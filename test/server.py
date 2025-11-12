# server.py (PATCHED for ImageFolder dataset, normalization, logging)
import os
import json
import time
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from torchvision import models
import torchvision.transforms as T
from torchvision.models import resnet34, ResNet34_Weights
from model import PokemonCNN
from dataset import load_pokemon_dataset
from utils import receive_data

# --- Configurações ---
HOST = '127.0.0.1'
PORT = 65432
MODEL_PATH = 'pokemon_model_best.pth'  # use the best checkpoint produced by train.py
IMG_SIZE = 224

# Dataset base path (ImageFolder style)
DATASET_PATH = '/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14'

# Hiperparâmetros do Ataque DLG
adam_lr = 0.005
adam_iters = 5000
lbfgs_iters = 3
lambda_tv_start = 5e-6
lambda_tv_end = 5e-6
lambda_l2 = 1e-4
use_cosine = False
save_every = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Servidor usando device: {device}")

# --- Debug / logging dir ---
DEBUG_OUT = "debug_outputs"
os.makedirs(DEBUG_OUT, exist_ok=True)
LOG_PATH = os.path.join(DEBUG_OUT, "attack_log.jsonl")

# simple JSONL logger
def log_event(obj):
    obj["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(obj, default=lambda o: o if isinstance(o, (int,float,str,bool)) else str(o)) + "\n")

# --- Utility functions ---
def total_variation(x):
    # x assumed shape (B,C,H,W)
    h = ((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2).sum()
    w = ((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2).sum()
    return h + w

def norm_grad_tensor(g):
    return g / (g.norm() + 1e-10)

def gradient_matching_loss_normalized(grads_pred, grads_true, use_cosine=True):
    mse = 0.0
    cosine = 0.0
    for gp, gt in zip(grads_pred, grads_true):
        gn_p = norm_grad_tensor(gp)
        gn_t = norm_grad_tensor(gt.to(gp.device))
        mse = mse + ((gp - gn_t) ** 2).mean()
        if use_cosine:
            cosine = cosine + (1.0 - torch.nn.functional.cosine_similarity(gn_p.view(-1), gn_t.view(-1), dim=0))
    return mse + (0.1 * cosine if use_cosine else 0.0)

# --- helpers for saving / debug ---

def save_tensor_img_raw(t, name):
    """Save a tensor in [0,1] range (C,H,W) or (B,C,H,W)."""
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    t = t.clamp(0, 1)
    save_image(t, os.path.join(DEBUG_OUT, name))

def save_grad_norm_map(x_hat, name="grad_norm.png"):
    if getattr(x_hat, "grad", None) is None:
        return
    gr = x_hat.grad.detach().cpu()
    if gr.ndim == 4:
        gr = gr[0]
    norm_map = gr.norm(dim=0).numpy()
    plt.imsave(os.path.join(DEBUG_OUT, name), norm_map)

# --- Load minimal info: classes + mean/std (server) ---
print("Lendo classes e mean/std para visualização e normalização...")

# 1) Tentar ler stats do payload (se disponível) ou do arquivo local
# Assuma que 'payload' pode existir mais tarde; por ora, use local file
stats_file = os.path.join(DEBUG_OUT, "dataset_stats.json")
if os.path.exists(stats_file):
    with open(stats_file, "r") as f:
        s = json.load(f)
        mean, std = s.get("mean"), s.get("std")
        if not (isinstance(mean, list) and len(mean) == 3 and isinstance(std, list) and len(std) == 3):
            raise RuntimeError(f"Formato inválido em {stats_file}: mean/std esperados como listas de 3 floats.")
else:
    raise FileNotFoundError(f"{stats_file} não encontrado. Rode train.py (ou peça ao cliente para enviar dataset_stats_path).")

# 2) Carregar classes (se precisar)
classes_file = os.path.join(DEBUG_OUT, "classes.json")
if os.path.exists(classes_file):
    with open(classes_file, "r") as f:
        classes = json.load(f).get("classes", [])
else:
    # fallback: carregar dataset apenas para recuperar classes (más performance)
    print("[WARN] classes.json não encontrado — carregando dataset para obter classes (mais lento)...")
    initial_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    train_loader_tmp, valid_loader_tmp, test_loader_tmp, classes_tmp, _, _ = load_pokemon_dataset(
        base_path=DATASET_PATH, batch_size=64, train_transform=initial_transform, test_transform=initial_transform
    )
    classes = classes_tmp

NUM_CLASSES = len(classes)
print(f"N classes (dataset): {NUM_CLASSES}")

# 3) Normalization helpers (idênticos ao train.py)
mean_t = torch.tensor(mean, device=device).view(1,3,1,1)
std_t = torch.tensor(std, device=device).view(1,3,1,1)

def normalize_tensor(x):
    return (x - mean_t) / (std_t + 1e-12)

def denormalize_tensor(x_norm):
    return x_norm * std_t + mean_t

# --- Preparar Modelo ---
try:
    print("Carregando modelo ResNet34 para reconstrução...")

    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[DEBUG] Missing keys:", missing)
    print("[DEBUG] Unexpected keys:", unexpected)
    print("Modelo ResNet34 carregado com sucesso.")
    criterion = nn.CrossEntropyLoss()
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    raise

# --- Loop Principal do Servidor ---
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Servidor escutando em {HOST}:{PORT}...")
    conn, addr = s.accept()
    with conn:
        print(f"Conexão recebida de {addr}")
        payload = None
        try:
            payload = receive_data(conn)
            if not isinstance(payload, dict):
                raise ValueError("Payload recebido tem formato inválido")
            print("[SERVER] Payload recebido: chaves:", list(payload.keys()))

            received_data = payload  # reutiliza o mesmo dicionário
        except Exception as e:
            print("[SERVER] Erro ao receber payload:", e)
            conn.close()
            # decide: continuar loop (esperar próximo cliente) ou abortar
            raise

        # received_data = receive_data(conn)
        # if received_data is None:
        #     print("Falha ao receber dados. Encerrando.")
        #     exit()

        target_grads = [torch.tensor(g).to(device) for g in received_data['gradients']]
        for i, (g, p) in enumerate(zip(target_grads, model.parameters())):
            if tuple(g.shape) != tuple(p.shape):
                raise ValueError(f"Gradient shape mismatch at param {i}: received {g.shape}, expected {tuple(p.shape)}")
        target_label_idx = received_data['label_idx']
        target_label_name = received_data.get('label_name', str(target_label_idx))
        secret_image_idx = received_data.get('secret_image_idx', 0)

        target_label = torch.tensor([target_label_idx], dtype=torch.long).to(device)

        print(f"Gradientes recebidos. Rótulo alvo: {target_label_idx} ({target_label_name})")
        print(f"Índice da imagem secreta: {secret_image_idx}")
        log_event({"event":"attack_start","label":target_label_idx,"label_name":target_label_name})

        # Preparar Imagem Dummy (iniciada no espaço [0,1] via inverse tanh trick)
        # We'll parametrize dummy_img in tanh-space but convert to [0,1] before normalizing to match model input
        dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device, dtype=torch.float32, requires_grad=True)

        def get_grads_from_model(x_01, y):
            # x_01: image in [0,1]
            x_norm = normalize_tensor(x_01)
            model.zero_grad()
            loss = criterion(model(x_norm), y)
            return torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # --- ADAM warm-up ---
        print("--- Otimização Adam (Warm-up) ---")
        optimizer = optim.Adam([dummy_img], lr=adam_lr)
        best_loss = float('inf')
        best_img_adam = None

        for i in range(adam_iters):
            optimizer.zero_grad()

            clipped_01 = (torch.tanh(dummy_img) + 1.0) / 2.0  # map to [0,1]
            grads_pred = get_grads_from_model(clipped_01, target_label)

            t = i / max(1, adam_iters - 1)
            lambda_tv = lambda_tv_start * (1 - t) + lambda_tv_end * t

            loss_g = gradient_matching_loss_normalized(grads_pred, target_grads, use_cosine)
            tv = total_variation(clipped_01)
            l2im = torch.mean(clipped_01 ** 2)
            loss = loss_g + lambda_tv * tv + lambda_l2 * l2im

            loss.backward()
            optimizer.step()

            if (i + 1) % save_every == 0:
                print(f"[Adam {i+1}/{adam_iters}] loss_g={loss_g.item():.6e} tv={tv.item():.6e} l2={l2im.item():.6e} total={loss.item():.6e}")
                log_event({"phase":"adam","iter":i+1,"loss_g":loss_g.item(),"tv":tv.item(),"l2":l2im.item()})

            if loss_g.item() < best_loss:
                best_loss = loss_g.item()
                best_img_adam = clipped_01.detach().clone()
                save_tensor_img_raw(best_img_adam, f"best_adam_iter{i+1}.png")
                log_event({"phase":"adam","iter":i+1,"best":True,"loss_g":best_loss,"saved":"best_adam_iter%d.png" % (i+1)})

        if best_img_adam is None:
            print("WARNING: best_img_adam is None after Adam. Using current clipped.")
            best_img_adam = (torch.tanh(dummy_img)).detach().clone()

        # --- Refinamento LBFGS ---
        print("--- Refinamento LBFGS ---")

        # initialize LBFGS parameter so that tanh(dummy_img) == (best_img_adam*2-1)
        with torch.no_grad():
            inv = torch.atanh(best_img_adam * 2.0 - 1.0)
            if inv.shape == dummy_img.shape:
                dummy_img.data.copy_(inv)
            else:
                try:
                    dummy_img.data.copy_(inv.view_as(dummy_img))
                except Exception as e:
                    print("[DEBUG] could not copy inverse tanh to dummy_img:", e)
                    try:
                        dummy_img.data.copy_(best_img_adam)
                    except Exception as e2:
                        print("[DEBUG] fallback copy also failed:", e2)

        dummy_img.requires_grad_(True)
        lbfgs_opt = torch.optim.LBFGS([dummy_img], max_iter=30, lr=0.01, history_size=50)
        globals()["current_lbfgs_iter"] = 0

        def lbfgs_closure():
            lbfgs_opt.zero_grad()
            clipped_01 = (torch.tanh(dummy_img) + 1.0) / 2.0
            grads_pred = get_grads_from_model(clipped_01, target_label)
            loss_g = gradient_matching_loss_normalized(grads_pred, target_grads, use_cosine)
            tv = total_variation(clipped_01)
            loss = loss_g + lambda_l2 * torch.mean(clipped_01 ** 2)
            loss.backward()

            it = globals().get("current_lbfgs_iter", 0)
            if it % 25 == 0:
                try:
                    save_tensor_img_raw(clipped_01.detach(), f"lbfgs_iter{it}.png")
                    if dummy_img.grad is not None:
                        save_grad_norm_map(dummy_img, f"grad_iter{it}.png")
                    print(f"[LBFGS closure] iter={it} loss_g={loss_g.item():.6e} tv={tv.item():.6e}")
                    log_event({"phase":"lbfgs","iter":it,"loss_g":loss_g.item(),"tv":tv.item()})
                except Exception as e:
                    print("[DEBUG] error in LBFGS closure logging:", e)
            globals()["current_lbfgs_iter"] = it + 1
            return loss

        for step in range(lbfgs_iters):
            try:
                ret = lbfgs_opt.step(lbfgs_closure)
                loss_total_val = ret.item() if isinstance(ret, torch.Tensor) else (float(ret) if isinstance(ret, (float, int)) else None)
            except Exception as e:
                print("[DEBUG] LBFGS step failed:", e)
                break
            if (step + 1) % 25 == 0:
                print(f"[LBFGS {step+1}/{lbfgs_iters}] approx_total={loss_total_val}")

        print("Ataque DLG (Adam + LBFGS) concluído.")
        final_clipped = (torch.tanh(dummy_img) + 1.0) / 2.0
        save_tensor_img_raw(final_clipped, "final_recon.png")
        log_event({"event":"attack_end","final_loss_g":float(best_loss)})
        print(f"[DEBUG] final reconstruction saved: {os.path.join(DEBUG_OUT, 'final_recon.png')}")

        # --- Carregar imagem original para comparação (robusto) ---
        print("Carregando imagem original para comparação...")

        from PIL import Image
        
        # payload pode ter 'secret_image_path' e 'secret_image_idx'
        secret_path = payload.get("secret_image_path", None) if isinstance(payload, dict) else None
        secret_idx = int(payload.get("secret_image_idx", 0)) if isinstance(payload, dict) and payload.get("secret_image_idx") is not None else 0

        # transform usado para visualização (sem normalização)
        vis_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

        img_true_vis = None

        # 1) Se o cliente enviou caminho absoluto da imagem, use isso (mais robusto)
        if secret_path:
            try:
                pil = Image.open(secret_path).convert("RGB")
                t = vis_transform(pil)  # tensor in [0,1], CxHxW
                img_true_vis = t.squeeze(0) if t.dim() == 4 else t  # C,H,W
                img_true_vis = img_true_vis.permute(1,2,0).numpy()   # H,W,C
                print(f"[SERVER] Carregada imagem original a partir de secret_image_path: {secret_path}")
            except Exception as e:
                print(f"[SERVER] Falha ao abrir secret_image_path ({secret_path}): {e}")
                img_true_vis = None

        # 2) Se não fornecido o path, tente usar valid_loader (se existir)
        if img_true_vis is None:
            try:
                # tentamos criar um loader mínimo (sem augment) se necessário
                train_loader_vis, valid_loader_vis, test_loader_vis, classes_vis, _, _ = load_pokemon_dataset(
                    base_path=DATASET_PATH,
                    batch_size=1,
                    train_transform=vis_transform,
                    test_transform=vis_transform,
                    auto_create_valid=True  # garante criação de valid se não existir
                )

                chosen_loader = valid_loader_vis if valid_loader_vis is not None else train_loader_vis
                if chosen_loader is None:
                    raise RuntimeError("Nenhum loader disponível após tentativa de criação automática.")
                
                vis_list = list(chosen_loader)  # lista de (img, label)
                if len(vis_list) == 0:
                    raise RuntimeError("Loader retornou lista vazia.")
                
                # se o loader for Subset, indices podem ser relativos ao dataset; mas list() já retornou exemplos na ordem de iteração
                # escolhemos o índice fornecido pelo cliente, se existir; caso contrário pegamos primeiro
                if secret_idx < len(vis_list):
                    img_tensor, _ = vis_list[secret_idx]
                else:
                    img_tensor, _ = vis_list[0]

                # img_tensor: shape (1, C, H, W) ou (C, H, W) dependendo do DataLoader; normalizamos para HWC numpy [0,1]
                if img_tensor.dim() == 4:
                    img_true_vis = img_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
                else:
                    img_true_vis = img_tensor.permute(1,2,0).cpu().numpy()

                print(f"[SERVER] Carregada imagem original via loader (len={len(vis_list)}), usando idx {secret_idx if secret_idx < len(vis_list) else 0}")

            except Exception as e:
                print(f"[SERVER] Falha ao carregar via loader: {e}")
                img_true_vis = None

        # 3) Se tudo falhar, tentar procurar o arquivo manualmente na árvore do dataset (fallback)
        if img_true_vis is None:
            try:
                # tenta deduzir caminho a partir do DATASET_PATH e das classes e índice
                classes_file = os.path.join("debug_outputs", "classes.json")
                if os.path.exists(classes_file):
                    cls_list = json.load(open(classes_file))["classes"]
                    # se secret_idx menor que n_classes, tentamos abrir primeira imagem da classe
                    if 0 <= secret_idx < len(cls_list):
                        candidate_dir = os.path.join(DATASET_PATH, "train", cls_list[secret_idx])
                        if os.path.isdir(candidate_dir):
                            # pega primeiro arquivo da pasta
                            files = [f for f in os.listdir(candidate_dir) if os.path.isfile(os.path.join(candidate_dir, f))]
                            if files:
                                path_try = os.path.join(candidate_dir, files[0])
                                pil = Image.open(path_try).convert("RGB")
                                t = vis_transform(pil)
                                img_true_vis = t.permute(1,2,0).cpu().numpy()
                                print(f"[SERVER] Fallback: carreguei {path_try}")
            except Exception as e:
                print(f"[SERVER] Fallback falhou: {e}")
                img_true_vis = None

        # Se ainda não conseguimos a imagem original, avise e prossiga apenas com a reconstrução
        if img_true_vis is None:
            print("[SERVER] Aviso: Não foi possível carregar a imagem original para comparação. Apenas a reconstrução será salva.")
        else:
            # img_recon: final_clipped (supondo dummy image já denormalizada e em [0,1])
            # certifique-se que 'final_clipped' está no device/cpu
            recon = final_clipped.detach().cpu().squeeze(0)   # C,H,W
            recon_hwc = recon.permute(1,2,0).numpy()         # H,W,C

            # Opcional: garantir valores em [0,1] e converter para uint8 ao salvar/mostrar
            recon_hwc = (np.clip(recon_hwc, 0.0, 1.0) * 255).astype('uint8')
            img_true_uint8 = None
            if img_true_vis is not None:
                img_true_uint8 = (np.clip(img_true_vis, 0.0, 1.0) * 255).astype('uint8')

            # salvar ambos lado a lado se disponível
            out_dir = "debug_outputs"
            os.makedirs(out_dir, exist_ok=True)
            from PIL import Image
            Image.fromarray(recon_hwc).save(os.path.join(out_dir, "final_recon.png"))
            print(f"[SERVER] final reconstruction saved: {os.path.join(out_dir, 'final_recon.png')}")
            if img_true_uint8 is not None:
                Image.fromarray(img_true_uint8).save(os.path.join(out_dir, "original_vis.png"))
                print(f"[SERVER] original image saved: {os.path.join(out_dir, 'original_vis.png')}")

