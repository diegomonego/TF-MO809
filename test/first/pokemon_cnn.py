# pokemon_cnn.py
import torch
import torch.nn as nn

class PokemonCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Imagem de entrada: 3 canais (RGB)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # -> 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # -> 16x16
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # -> 8x8
        )
        # Dimensão calculada: 256 canais * 8 * 8 = 16384
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, num_classes) 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

if __name__ == '__main__':
    # Teste para garantir as dimensões
    model = PokemonCNN()
    # 1 imagem, 3 canais, 64x64
    test_input = torch.randn(1, 3, 64, 64) 
    output = model(test_input)
    print(f"Dimensão de saída: {output.shape} (deve ser [1, 1000])")