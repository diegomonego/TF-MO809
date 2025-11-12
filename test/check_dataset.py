# check_dataset.py
import os
from collections import defaultdict

BASE = "/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14/train"

counts = {}
for cls in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, cls)
    if os.path.isdir(p):
        n = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f))])
        counts[cls] = n

# print classes com <5 imagens
small = {c:n for c,n in counts.items() if n < 5}
print(f"Total classes: {len(counts)}")
if small:
    print("Classes com menos de 5 imagens (corrija ou remova):")
    for c,n in small.items():
        print(f" - {c}: {n}")
else:
    print("OK — todas as classes têm pelo menos 5 imagens.")

