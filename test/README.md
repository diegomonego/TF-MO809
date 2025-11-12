# README — Test 


## 1 — Visão geral rápida

Este diretório contém código para treinar/avaliar um classificador de Pokémon (ResNet34), prever imagens, gerar matriz de confusão e executar um ataque DLG (Deep Leakage from Gradients) em que um cliente envia gradientes e o servidor tenta reconstruir a imagem original a partir desses gradientes.

**Funcionalidades principais:**

* `train.py` — treina / fine-tunea uma ResNet34 no dataset Roboflow Pokedex v14 (pastas `train/` `valid/` `test/`).
* `predict.py` — classifica uma imagem (script de inferência rápido).
* `eval.py` — avalia no test set e gera matriz de confusão.
* `client.py` — simula o client que seleciona uma imagem (baixa perda), calcula gradientes e envia payload ao servidor.
* `server.py` — recebe payload, executa o ataque (Adam warm-up + LBFGS) e salva `final_recon.png`.
* `dataset.py` — utilitários para carregar o dataset ImageFolder (gera `debug_outputs/dataset_stats.json` e `classes.json`).
* `utils.py` — utilitários de envio/recebimento via socket (serialização).
* `model.py` — (se existir) modelo customizado (a pasta pode conter alternativa à ResNet).
* `debug_outputs/` — arquivos gerados: `dataset_stats.json`, `classes.json`, imagens intermediárias, `attack_log.jsonl`, checkpoints.

## 2 — Pré-requisitos / Ambiente

Recomendado: Python 3.10+, CUDA (opcional), ambiente isolado (`micromamba`/`venv`).

**Dependências principais (instalar via pip / conda):**

* `torch`, `torchvision`
* `pillow` (PIL)
* `numpy`
* `matplotlib`
* `tensorboard` (opcional)
* `requests` (só se usar scripts auxiliares)
* outras libs utilitárias (ver `requirements.txt` se disponível)

**Exemplo (pip):**

```bash
pip install torch torchvision pillow numpy matplotlib tensorboard requests