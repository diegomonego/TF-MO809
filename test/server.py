# server.py (VERSÃO FINAL OTIMIZADA + DEBUG)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import socket
from model import PokemonCNN
from utils import receive_data
import torchvision.transforms as T
from dataset import PokemonCSVDataset
from torchvision.utils import save_image

# --- Configurações ---
HOST = '127.0.0.1'
PORT = 65432
MODEL_PATH = 'pokemon_model.pth'
IMG_SIZE = 64

# --- ATENÇÃO: Caminhos ---
CSV_PATH = '../DATASETS/Pokemon.csv'
IMAGE_DIR = '../DATASETS/all_pokemon/'

# --- Hiperparâmetros do Ataque DLG (CORRIGIDOS/OTIMIZADOS) ---
adam_lr = 0.01
adam_iters = 20000
lbfgs_iters = 250
lambda_tv_start = 1e-5
lambda_tv_end = 1e-6
lambda_l2 = 1e-10
use_cosine = True
save_every = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Servidor usando device: {device}")

# --- Debug output dir ---
DEBUG_OUT = "debug_outputs"
os.makedirs(DEBUG_OUT, exist_ok=True)

# --- Pré-carregar dataset (permanece igual) ---
print("Carregando dataset para obter metadados...")
try:
    temp_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    temp_dataset = PokemonCSVDataset(csv_file=CSV_PATH, img_dir=IMAGE_DIR, transform=temp_transform)
    NUM_CLASSES = len(temp_dataset.classes)
    print(f"Número de classes detectado: {NUM_CLASSES}")
    del temp_dataset
except FileNotFoundError:
    print(f"ERRO: Não foi possível encontrar os arquivos do dataset em {CSV_PATH} ou {IMAGE_DIR}")
    exit()
except Exception as e:
    print(f"ERRO ao carregar dataset temporário: {e}. Verifique os nomes das colunas no dataset.py")
    exit()

# --- Funções Utilitárias ---
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

def denormalize(img_tensor):
    # Inverse of Normalize(mean=0.5,std=0.5) used in training
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    denorm = T.Normalize(mean=[-m/s for m,s in zip(mean,std)], std=[1/s for s in std])
    return denorm(img_tensor)

# --- Debug helpers ---
def save_tensor_img(t, name):
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    t = t.clamp(-1.0, 1.0)  # recebe tanh-space
    # convert to [0,1] for saving: map [-1,1] -> [0,1]
    arr = (t + 1.0) / 2.0
    if arr.shape[0] in (1, 3):
        save_image(arr, os.path.join(DEBUG_OUT, name))
    else:
        # if last dim is channels
        a = arr.numpy()
        if a.ndim == 3 and a.shape[2] in (1,3):
            save_image(torch.from_numpy(a.transpose(2,0,1)), os.path.join(DEBUG_OUT, name))
        else:
            # fallback: save per-channel
            for i in range(a.shape[0]):
                save_image(torch.from_numpy(a[i:i+1]), os.path.join(DEBUG_OUT, f"{name}_c{i}.png"))

def log_shapes_and_stats(tensor, tag="tensor"):
    t = tensor.detach().cpu()
    try:
        print(f"[DEBUG] {tag} shape={tuple(t.shape)} dtype={t.dtype} min={t.min().item():.6f} max={t.max().item():.6f}")
    except Exception as e:
        print("[DEBUG] error printing stats:", e)

def save_grad_norm_map(x_hat, name="grad_norm.png"):
    if getattr(x_hat, "grad", None) is None:
        print("[DEBUG] x_hat.grad is None")
        return
    gr = x_hat.grad.detach().cpu()
    if gr.ndim == 4:
        gr = gr[0]
    norm_map = gr.norm(dim=0).numpy()
    plt.imsave(os.path.join(DEBUG_OUT, name), norm_map)
    print(f"[DEBUG] grad norm map saved: {os.path.join(DEBUG_OUT, name)}")

def quick_permute_checks(tensor, name="x_hat"):
    t = tensor.detach().cpu()
    print(f"[DEBUG] running permute plausibility checks for {name}")
    for perm in [(0,1,2),(1,2,0),(2,0,1),(2,1,0)]:
        try:
            cand = t.permute(perm).numpy()
            if cand.ndim == 3 and cand.shape[2] in (1,3):
                print(f"[DEBUG] perm {perm} -> plausible color last dim {cand.shape}")
        except Exception:
            pass

def atanh_tensor(x, eps=1e-6):
    # safe atanh approximate: 0.5 * (log1p(x) - log1p(-x))
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# --- Preparar Modelo (permanece igual) ---
try:
    model = PokemonCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    print("Modelo treinado (pokemon_model.pth) carregado com sucesso.")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    exit()

# --- Loop Principal do Servidor ---
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Servidor escutando em {HOST}:{PORT}...")
    conn, addr = s.accept()

    with conn:
        print(f"Conexão recebida de {addr}")

        received_data = receive_data(conn)
        if received_data is None:
            print("Falha ao receber dados. Encerrando.")
            exit()

        target_grads = received_data['gradients']
        target_label_idx = received_data['label_idx']
        target_label_name = received_data.get('label_name', str(target_label_idx))
        secret_image_idx = received_data['secret_image_idx']

        target_label = torch.tensor([target_label_idx], dtype=torch.long).to(device)

        print(f"Gradientes recebidos. Rótulo alvo: {target_label_idx} ({target_label_name})")
        print(f"Índice da imagem secreta: {secret_image_idx}")
        print("Iniciando ataque DLG...")

        # Preparar Imagem Dummy
        dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device, dtype=torch.float32, requires_grad=True)

        def get_grads(x, y):
            model.zero_grad()
            loss = criterion(model(x), y)
            return torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # --- ADAM warm-up ---
        print("\n--- Otimização Adam (Warm-up) ---")
        optimizer = optim.Adam([dummy_img], lr=adam_lr)
        best_loss = float('inf')
        best_img_adam = None

        for i in range(adam_iters):
            optimizer.zero_grad()

            clipped = torch.tanh(dummy_img)  # in [-1,1]
            grads_pred = get_grads(clipped, target_label)

            t = i / max(1, adam_iters - 1)
            lambda_tv = lambda_tv_start * (1 - t) + lambda_tv_end * t

            loss_g = gradient_matching_loss_normalized(grads_pred, target_grads, use_cosine)
            tv = total_variation(clipped)
            l2im = torch.mean(clipped ** 2)
            loss = loss_g + lambda_tv * tv + lambda_l2 * l2im

            loss.backward()
            optimizer.step()

            if (i + 1) % save_every == 0:
                print(f"[Adam {i+1}/{adam_iters}] loss_g={loss_g.item():.6e} tv={tv.item():.6e} l2={l2im.item():.6e} total={loss.item():.6e}")

            if loss_g.item() < best_loss:
                best_loss = loss_g.item()
                best_img_adam = clipped.detach().clone()
                # save immediate checkpoint to debug
                save_tensor_img(best_img_adam, f"best_adam_iter{i+1}.png")
                print(f"[DEBUG] Novo best_adam iter {i+1}, loss_g={best_loss:.6e} -> saved best_adam_iter{i+1}.png")

        if best_img_adam is None:
            print("WARNING: best_img_adam is None after Adam. Using current clipped.")
            best_img_adam = torch.tanh(dummy_img).detach().clone()

        # --- Refinamento LBFGS ---
        print("\n--- Refinamento LBFGS ---")

        # initialize LBFGS parameter so that tanh(dummy_img) == best_img_adam
        with torch.no_grad():
            # inverse tanh
            inv = atanh_tensor(best_img_adam)
            # ensure same shape
            if inv.shape == dummy_img.shape:
                dummy_img.data.copy_(inv)
            else:
                # try to adapt permutations if shapes mismatch (still attempt)
                try:
                    dummy_img.data.copy_(inv.view_as(dummy_img))
                except Exception as e:
                    print("[DEBUG] could not copy inverse tanh to dummy_img:", e)
                    # fallback: copy directly (may be double tanh but we try)
                    try:
                        dummy_img.data.copy_(best_img_adam)
                    except Exception as e2:
                        print("[DEBUG] fallback copy also failed:", e2)

        dummy_img.requires_grad_(True)

        lbfgs_opt = torch.optim.LBFGS([dummy_img], max_iter=20, lr=0.001, history_size=10)

        # closure state for logging
        globals()["current_lbfgs_iter"] = 0

        def lbfgs_closure():
            lbfgs_opt.zero_grad()

            clipped = torch.tanh(dummy_img)

            grads_pred = get_grads(clipped, target_label)
            loss_g = gradient_matching_loss_normalized(grads_pred, target_grads, use_cosine)
            tv = total_variation(clipped)
            loss = loss_g + lambda_l2 * torch.mean(clipped ** 2)  # TV intentionally removed here

            loss.backward()

            # periodic debug saves inside closure (lightweight)
            it = globals().get("current_lbfgs_iter", 0)
            if it % 25 == 0:
                try:
                    save_tensor_img(clipped.detach(), f"lbfgs_iter{it}.png")
                    # grad map (if produced)
                    if dummy_img.grad is not None:
                        save_grad_norm_map(dummy_img, f"grad_iter{it}.png")
                    print(f"[LBFGS closure] iter={it} loss_g={loss_g.item():.6e} tv={tv.item():.6e}")
                except Exception as e:
                    print("[DEBUG] error in LBFGS closure logging:", e)
            globals()["current_lbfgs_iter"] = it + 1
            return loss

        for step in range(lbfgs_iters):
            try:
                ret = lbfgs_opt.step(lbfgs_closure)
                # lbfgs_opt.step may return None or the loss; guard it
                loss_total_val = ret.item() if isinstance(ret, torch.Tensor) else (float(ret) if isinstance(ret, (float, int)) else None)
            except Exception as e:
                print("[DEBUG] LBFGS step failed:", e)
                break

            if (step + 1) % 25 == 0:
                print(f"[LBFGS {step+1}/{lbfgs_iters}] approx_total={loss_total_val}")

        print("Ataque DLG (Adam + LBFGS) concluído.")

        # Pega a imagem final otimizada
        final_clipped = torch.tanh(dummy_img).detach().cpu()
        save_tensor_img(final_clipped, "final_recon.png")
        print(f"[DEBUG] final reconstruction saved: {os.path.join(DEBUG_OUT, 'final_recon.png')}")

        print("Carregando imagem original para comparação...")

        transform_vis = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
        vis_dataset = PokemonCSVDataset(csv_file=CSV_PATH, img_dir=IMAGE_DIR, transform=transform_vis)

        img_true_vis, _ = vis_dataset[secret_image_idx]
        img_true_vis = img_true_vis.unsqueeze(0)

        # recon: final_clipped is in tanh-space [-1,1], denormalize expects normalized input used in training
        img_recon = denormalize(final_clipped.squeeze(0))
        # map [-1,1] -> [0,1] before plotting (denormalize may already scale)
        img_recon = (img_recon + 1.0) / 2.0
        img_recon = img_recon.permute(1, 2, 0).numpy()

        img_true_vis = img_true_vis.squeeze(0).permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(img_true_vis, 0, 1))
        plt.title(f"Original (Classe: {target_label_name})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(img_recon, 0, 1))
        plt.title("Reconstruída pelo DLG (Adam + LBFGS)")
        plt.axis('off')

        plt.suptitle("Resultado do Ataque de Inversão de Gradiente")
        plt.show()
