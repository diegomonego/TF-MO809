# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class PokemonCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Criar o índice de classes (ex: 'Pikachu' -> 25)
        self.classes = self.data_frame['name'].unique()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Criar um mapeamento reverso (ex: 25 -> 'Pikachu')
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_row = self.data_frame.iloc[idx]
        
        # Pega o nome do Pokémon (Rótulo)
        pokemon_name = img_row['name'] # Ajuste 'name' se a coluna for outra
        label = self.class_to_idx[pokemon_name]
        
        # Pega o nome do arquivo da imagem
        try:
            # Assumindo que os arquivos são 'NUMERO.jpg' e a coluna é '#'
            img_filename = str(img_row['number']) + '.jpg'
        except KeyError:
            print("ERRO: Verifique o nome da coluna no CSV para o nome do arquivo ('#')")
            return None, None
            
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Tenta com o nome do arquivo "limpo" (ex: 100.jpg vs 100)
            img_filename_alt = str(img_row['number']).split('.')[0] + '.jpg'
            img_path = os.path.join(self.img_dir, img_filename_alt)
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f"AVISO: Imagem não encontrada em {img_path}. Pulando.")
                return self.__getitem__((idx + 1) % len(self)) 

        if self.transform:
            image = self.transform(image)
            
        return image, label