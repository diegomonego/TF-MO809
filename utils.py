"""
Funções utilitárias e definições de modelos para tarefas de aprendizado profundo.

Este módulo fornece várias funções auxiliares e arquiteturas de redes neurais para
trabalhar com conjuntos de dados de imagens comuns como MNIST, CIFAR10, CIFAR100, FashionMNIST e CelebA.

Funcionalidades principais:
- Criação e pré-processamento de datasets
- Inicialização de modelos e pesos
- Manipulação de rótulos (codificação one-hot)
- Visualização e comparação de imagens
- Poda de modelos
- Definições de funções de perda

O módulo contém duas arquiteturas principais de modelo:
- MNISTCNN: Uma arquitetura CNN otimizada para datasets tipo MNIST (28x28 em escala de cinza)
- LeNet: Uma arquitetura LeNet modificada para datasets tipo CIFAR (32x32 RGB)

Todos os modelos e tensores podem ser executados tanto em CPU quanto em GPU dependendo da disponibilidade.
"""

import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import torch.nn.utils.prune as prune

# torch.manual_seed(50)

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando o dispositivo:", device)

def cria_dataset(dataset_name):
    """
    Cria e retorna um dataset e seu pipeline de transformação correspondente.
    
    Args:
        dataset_name (str): Nome do dataset ('mnist', 'cifar10', 'cifar100', 'fashionmnist', 'celeba')
        
    Returns:
        tuple: (dataset, pipeline_transformacao)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(28), 
                                 transforms.CenterCrop(28), 
                                 transforms.ToTensor()
                    ])
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(32), 
                                 transforms.CenterCrop(32), 
                                 transforms.ToTensor()
                    ])
    if dataset_name == 'cifar100':  
        dataset = datasets.CIFAR100(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(32), 
                                 transforms.CenterCrop(32), 
                                 transforms.ToTensor()
                    ])
    if dataset_name == 'fashionmnist':
        dataset = datasets.FashionMNIST(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(28), 
                                 transforms.CenterCrop(28), 
                                 transforms.ToTensor()
                    ])
    
    if dataset_name == 'celeba':
        dataset = datasets.CelebA(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(32), 
                                 transforms.CenterCrop(32), 
                                 transforms.ToTensor()
                    ])
    
    return dataset, tp

def weights_init(m):
    """
    Inicializa pesos e vieses de uma camada do modelo uniformemente entre -0.5 e 0.5.
    
    Args:
        m (nn.Module): Camada do modelo a ser inicializada
    """
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
        
def weights_init_mnist(m):
    """
    Inicializa pesos e vieses de uma camada do modelo uniformemente entre 0 e 1.
    Especificamente projetado para modelos MNIST.
    
    Args:
        m (nn.Module): Camada do modelo a ser inicializada
    """
    if hasattr(m, "weight"):
        m.weight.data.uniform_(0, 1)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(0, 1)

def cria_modelo(dataset_name):
    """
    Cria e retorna uma arquitetura de modelo apropriada para o dataset fornecido.
    
    Args:
        dataset_name (str): Nome do dataset para criar o modelo
        
    Returns:
        nn.Module: Modelo inicializado no dispositivo apropriado
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        modelo = MNISTCNN().to(device)
        
    if dataset_name == 'cifar10':
        modelo = LeNet(num_classes=10).to(device)
        modelo.apply(weights_init)
        
    if dataset_name == 'cifar100':
        modelo = LeNet(num_classes=100).to(device)
        modelo.apply(weights_init)
        
    if dataset_name == 'fashionmnist':
        modelo = MNISTCNN().to(device)
        
    return modelo
    
def poda_modelo(model, amount=0.1):
    """
    Aplica poda não estruturada aleatória às camadas convolucionais e lineares.
    
    Args:
        model (nn.Module): Modelo a ser podado
        amount (float): Fração de parâmetros a serem podados (padrão: 0.1)
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.random_unstructured(module, name="weight", amount=amount)
            print(f"Podando Layer {name}")

def label_to_onehot(target, num_classes=10):
    """
    Converte rótulos inteiros para formato one-hot.
    
    Args:
        target (torch.Tensor): Rótulos inteiros
        num_classes (int): Número de classes (padrão: 10)
        
    Returns:
        torch.Tensor: Rótulos codificados em one-hot
    """
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def get_imagem_e_label(img_index, dataset, tp, num_classes=10):
    """
    Recupera e pré-processa uma imagem e seu rótulo de um dataset.
    
    Args:
        img_index (int): Índice da imagem no dataset
        dataset (Dataset): Dataset para obter a imagem
        tp (transforms.Compose): Pipeline de transformação
        num_classes (int): Número de classes para codificação one-hot
        
    Returns:
        tuple: (imagem_preprocessada, rotulo_one_hot)
    """
    gt_data         = tp(dataset[img_index][0]).to(device)
    gt_data         = gt_data.view(1, *gt_data.size())
    gt_label        = torch.Tensor([dataset[img_index][1]]).long().to(device)
    gt_label        = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)
    
    return gt_data, gt_onehot_label

def cria_ruido(tamanho_imagem, tamanho_label):
    """
    Cria tensores de ruído aleatório para imagem e rótulo com gradientes habilitados.
    
    Args:
        tamanho_imagem (tuple): Forma do tensor da imagem
        tamanho_label (tuple): Forma do tensor do rótulo
        
    Returns:
        tuple: (ruido_imagem, ruido_rotulo)
    """
    fake_data  = torch.randn(tamanho_imagem).to(device).requires_grad_(True)
    fake_label = torch.randn(tamanho_label).to(device).requires_grad_(True)
    
    return fake_data, fake_label

def mostra_imagem(imagem):
    """
    Exibe uma única imagem usando matplotlib.
    
    Args:
        imagem (torch.Tensor): Tensor da imagem a ser exibida
    """
    plt.figure(figsize=(2, 2))
    tt = transforms.ToPILImage()
    plt.imshow(tt(imagem[0].cpu()))
    
def cross_entropy_for_onehot(pred, target):
    """
    Calcula a perda de entropia cruzada para alvos codificados em one-hot.
    
    Args:
        pred (torch.Tensor): Previsões do modelo
        target (torch.Tensor): Rótulos alvo codificados em one-hot
        
    Returns:
        torch.Tensor: Perda de entropia cruzada
    """
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def compara_imagens(imagem_real, imagem_falsa):
    """
    Exibe duas imagens lado a lado para comparação.
    
    Args:
        imagem_real (torch.Tensor): Primeira imagem a ser exibida
        imagem_falsa (torch.Tensor): Segunda imagem a ser exibida
    """
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    tt = transforms.ToPILImage()
    axs[0].imshow(tt(imagem_real[0].cpu()))
    axs[1].imshow(tt(imagem_falsa[0].cpu()))
    plt.show()

def plot_historico_ataque(historico, label_fake):
    """
    Plota uma grade de imagens mostrando a progressão de um ataque.
    
    Args:
        historico (list): Lista de imagens mostrando a progressão do ataque
        label_fake (torch.Tensor): Rótulo previsto para o ataque
    """
    tt = transforms.ToPILImage()
    plt.figure(figsize=(15, 3.5))
    for i in range(14):
        plt.subplot(2, 7, i + 1)
        plt.imshow(historico[i * 10])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    print("Label predito é %d." % torch.argmax(label_fake, dim=-1).item())
    
class MNISTCNN(nn.Module):
    """
    Arquitetura CNN otimizada para datasets tipo MNIST (imagens em escala de cinza 28x28).
    
    Arquitetura:
    - 2 camadas convolucionais com ReLU e max pooling
    - 2 camadas totalmente conectadas
    """
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.contiguous().view(out.size(0), -1)  # Adiciona .contiguous()
        out = self.fc(out)
        return out

class LeNet(nn.Module):
    """
    Arquitetura LeNet modificada para datasets tipo CIFAR (imagens RGB 32x32).
    
    Arquitetura:
    - 4 camadas convolucionais com ativação Sigmoid
    - 1 camada totalmente conectada
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.contiguous().view(out.size(0), -1)  # Adiciona .contiguous()
        out = self.fc(out)
        return out