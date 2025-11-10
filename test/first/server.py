# server.py
import torch
import torch.nn as nn
import socket
import matplotlib.pyplot as plt
from pokemon_cnn import PokemonCNN
from utils import receive_data # Importa utilitário de recebimento

# --- Configurações de Comunicação ---
HOST = '127.0.0.1'
PORT = 65432

# --- Setup do Ataque DLG (Funções e Utilitários) ---
# Essas funções são as mesmas que você usou no seu código inicial (otimização DLG + LBFGS)
def total_variation(x):
    # Cálculo de Total Variation (regularização de suavidade)
    h = ((x[:,:,1:,:] - x[:,:,:-1,:])**2).sum()
    w = ((x[:,:,:,1:] - x[:,:,:,:-1])**2).sum()
    return h + w

def norm_grad_tensor(g):
    return g / (g.norm() + 1e-10)

def gradient_matching_loss_normalized(grads_pred, grads_true, use_cosine=True):
    mse = 0.0
    cosine = 0.0
    for gp, gt in zip(grads_pred, grads_true):
        gn_p = norm_grad_tensor(gp)
        gn_t = norm_grad_tensor(gt)
        mse = mse + ((gp - gt.to(gp.device))**2).mean()
        if use_cosine:
            cosine = cosine + (1.0 - torch.nn.functional.cosine_similarity(gn_p.view(-1), gn_t.view(-1), dim=0))
    return mse + (0.1 * cosine if use_cosine else 0.0)

# --- Setup do Modelo e Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PokemonCNN(num_classes=1000).to(device)
criterion = nn.CrossEntropyLoss()
model.eval()

# 1. Carregar Pesos Treinados (Essencial para o ataque!)
try:
    model.load_state_dict(torch.load('pokemon_cnn.pt', map_location=device))
    print("Pesos do modelo carregados com sucesso para o ataque.")
except:
    print("AVISO: Modelo não treinado (pokemon_cnn.pt não encontrado). O ataque pode falhar.")

# --- Lógica do Servidor ---
print(f"Servidor DLG iniciando em {HOST}:{PORT}...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Aguardando conexão do cliente...")

    conn, addr = s.accept()
    with conn:
        print(f"Conectado por {addr}")

        # 2. Receber Gradientes do Cliente
        target_grads = receive_data(conn)
        if target_grads is None:
            print("Não foi possível receber os gradientes.")
            return

        print(f"Gradientes recebidos. Iniciando ataque DLG...")

        # 3. Inferir o Rótulo (iDLG) - Opcional, mas útil
        # O gradiente do último layer (output) costuma ter a maior magnitude no índice do rótulo real
        inferred_label = torch.argmin(target_grads[-1]).item()
        target_label = torch.tensor([inferred_label], dtype=torch.long).to(device)
        print(f"Rótulo inferido (iDLG): {inferred_label}")

        # 4. Configuração do Ataque DLG
        img_size = (1, 3, 64, 64) # Batch, Canais, Altura, Largura
        dummy_img = torch.rand(img_size).to(device).requires_grad_(True)

        # Hiperparâmetros de Otimização (Ajuste fino será crucial!)
        adam_lr = 0.1
        adam_iters = 5000 # Reduzido para teste. Aumente para 10000+
        lambda_tv = 1e-4
        lambda_l2 = 1e-6

        adam_opt = torch.optim.Adam([dummy_img], lr=adam_lr)

        def get_grads(x, y):
            model.zero_grad()
            loss = criterion(model(x), y)
            # Retorna o gradiente da loss em relação aos pesos do modelo
            return torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # 5. Otimização (DLG)
        print("\n--- Ataque DLG (Adam Warm-up) ---")
        for i in range(adam_iters):
            adam_opt.zero_grad()
            clipped = dummy_img.clamp(0,1)
            grads_pred = get_grads(clipped, target_label)

            # Loss principal + Regularização
            loss_g = gradient_matching_loss_normalized(grads_pred, target_grads)
            tv = total_variation(clipped)
            loss = loss_g + lambda_tv * tv + lambda_l2 * torch.mean(clipped**2)

            loss.backward()
            adam_opt.step()
            with torch.no_grad():
                dummy_img.data.clamp_(0,1)

            if (i+1) % 500 == 0:
                print(f"[Adam {i+1}/{adam_iters}] loss_g={loss_g.item():.6e} total={loss.item():.6e}")

        # 6. Visualização do Resultado
        img_recon = dummy_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()

        plt.figure(figsize=(5,5))
        plt.imshow(img_recon)
        plt.title(f"Reconstrução DLG (Rótulo Inferido: {inferred_label})")
        plt.axis('off')
        plt.show()

        print("Ataque DLG finalizado e imagem reconstruída exibida.")
