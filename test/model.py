# model.py (Versão Otimizada)
import torch
import torch.nn as nn

class PokemonCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Entrada esperada: (Batch, 3, 64, 64)
        self.encoder = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),                         # ADIÇÃO: Normalização de Batch
            nn.ReLU(),
            nn.MaxPool2d(2),                            # (32, 32, 32)

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, 32, 32)
            nn.BatchNorm2d(64),                         # ADIÇÃO: Normalização de Batch
            nn.ReLU(),
            nn.MaxPool2d(2),                            # (64, 16, 16)

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (128, 16, 16)
            nn.BatchNorm2d(128),                        # ADIÇÃO: Normalização de Batch
            nn.ReLU(),
            nn.MaxPool2d(2),                            # (128, 8, 8)
        )

        # A dimensão do flatten será 128 * 8 * 8 = 8192
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),                            # ADIÇÃO: Dropout (Regularização)
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x