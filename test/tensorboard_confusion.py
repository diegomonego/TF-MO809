#!/usr/bin/env python3
"""
tensorboard_and_confusion.py
Utilitários para Logging no TensorBoard durante treino e para salvar a confusion matrix.


Uso (exemplo curto):
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment1')
writer.add_scalar('train/loss', loss, epoch)
writer.add_scalar('val/acc', val_acc, epoch)
writer.close()


Este arquivo contém também a função `log_confusion_matrix` que converte a matriz para imagem e envia ao TensorBoard.
"""
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix




def plot_confusion_matrix_figure(cm, class_names, normalize=False):
    fig = plt.figure(figsize=(10, 10))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black', fontsize=5)
    plt.tight_layout()
    return fig




def figure_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.array(img)
    return arr




def log_confusion_matrix_tb(writer, cm, class_names, tag='confusion', step=0, normalize=False):
    fig = plot_confusion_matrix_figure(cm, class_names, normalize=normalize)
    img = figure_to_image(fig)
    # writer is a SummaryWriter
    writer.add_image(tag, img.transpose(2, 0, 1), global_step=step)