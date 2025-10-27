import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss_acc(FILENAME):
    df_train = pd.read_csv(f'{FILENAME}-train.csv', names=['server_round', 'cid', 'acc', 'loss'])
    df_test  = pd.read_csv(f'{FILENAME}-evaluate.csv', names=['server_round', 'cid', 'acc', 'loss'])

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    

    sns.lineplot(data=df_train, x='server_round', y='loss', ax=ax[0], color='b', label='Loss Treino')
    sns.lineplot(data=df_train, x='server_round', y='acc', ax=ax[1], color='b', label='Acc Treino')
    sns.lineplot(data=df_test, x='server_round', y='loss', ax=ax[0], color='r', label='Loss Teste')
    sns.lineplot(data=df_test, x='server_round', y='acc', ax=ax[1], color='r', label='Acc Teste')

    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')

    ax[0].grid(True, linestyle=':')
    ax[1].grid(True, linestyle=':')

def plot_performance_distribution(FILENAME, NCLIENTS):
    
    df_train = pd.read_csv(f'{FILENAME}-train.csv', names=['server_round', 'cid', 'acc', 'loss'])
    df_test  = pd.read_csv(f'{FILENAME}-evaluate.csv', names=['server_round', 'cid', 'acc', 'loss'])
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(x=df_test['acc'].values[-NCLIENTS:], kde=True, color='r', bins=10, ax=ax[0])
    sns.barplot(x=df_test['cid'].values[-NCLIENTS:], y=df_test['acc'].values[-NCLIENTS:], color='b', ec='k', ax=ax[1])
    
    ax[0].set_title('Distribuição de Acurácia dos Clientes')
    ax[0].set_ylabel('Quantidade de Clientes')
    ax[0].set_xlabel('Acurácia Teste(%)')
    
    ax[1].set_title('Acurácia por Cliente')
    ax[1].set_ylabel('Acurácia Teste')
    ax[1].set_xlabel('Client ID (#)')
    
    for _ in range(2):
        ax[_].grid(True, linestyle=':')
        ax[_].set_axisbelow(True)