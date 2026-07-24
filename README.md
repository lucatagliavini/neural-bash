# neural-bash — Neural Network in AWK/Bash

Framework completo per il training e l'inferenza di reti neurali, implementato interamente in AWK con wrapper Bash. Nessuna dipendenza esterna oltre a `gawk`.

## Struttura del Progetto

```
.
├── lib/framework/
│   ├── utils-math.awk          # Inizializzazione pesi (Xavier, He, uniform), Gaussian
│   ├── utils-activation.awk    # sigmoid, tanh, relu, leaky_relu, softmax
│   ├── utils-shared.awk        # Logging, I/O dataset, normalizzazione z-score, metriche
│   ├── utils-network.awk       # Caricamento/salvataggio modello
│   ├── utils-forward.awk       # Forward pass (con dropout)
│   ├── utils-backward.awk      # Backpropagation
│   ├── utils-update.awk        # SGD, Momentum, Adam, gradient clipping, weight decay
│   ├── utils-loss.awk          # MSE, Cross-Entropy
│   ├── nnet-train.awk          # Loop di training (mini-batch, early stopping, checkpoint)
│   └── nnet-predict.awk        # Inferenza e metriche
├── dataset/
│   ├── xor.txt                 # XOR (2 input, 1 output, 4 campioni)
│   ├── and.txt                 # AND
│   ├── or.txt                  # OR
│   ├── iris_toy.txt            # Iris subset (4 input, 3 classi one-hot, 45 campioni)
│   └── sine_regression.txt     # sin(2πx) (1 input, 1 output, 30 campioni)
├── models/
│   └── xor/                    # Modello XOR pre-addestrato
├── tests/
│   ├── run.sh                  # Orchestratore test suite (222 test)
│   └── test_*.sh               # Dispatcher per categoria
├── nnet-init.sh                # Inizializzazione rete
└── nnet-run.sh                 # Train / predict / eval
```

## Quick Start

```bash
# 1. Inizializza una rete 2→8→1
./nnet-init.sh models/xor 2,8,1 --activation sigmoid --method xavier

# 2. Addestra
./nnet-run.sh train dataset/xor.txt models/xor --epochs 3000 --lr 0.5

# 3. Predici
./nnet-run.sh predict dataset/xor.txt models/xor

# 4. Addestra e valuta in un solo comando
./nnet-run.sh eval dataset/xor.txt models/xor --epochs 3000
```

## nnet-init.sh

```bash
./nnet-init.sh <model_dir> <architettura> [opzioni]
```

| Opzione | Default | Descrizione |
|---|---|---|
| `--activation FUNC` | `sigmoid` | Attivazione per tutti i layer |
| `--hidden-act FUNC` | (uguale a `--activation`) | Attivazione solo per i layer hidden; utile per separare hidden da output (es. sigmoid hidden + softmax output) |
| `--method METHOD` | `xavier` | Metodo inizializzazione: `xavier`, `he`, `random` |
| `--seed N` | (casuale) | Seed per riproducibilità |
| `--force` | — | Sovrascrive senza chiedere conferma |

**Attivazioni disponibili:** `sigmoid`, `tanh`, `relu`, `leaky_relu[:alpha]`, `softmax`, `linear`

**Linee guida inizializzazione:**
- `xavier` → con sigmoid / tanh
- `he` → con relu / leaky_relu
- `softmax` → solo layer di output per classificazione multi-classe

```bash
# Classificazione binaria
./nnet-init.sh models/xor 2,8,1 --activation sigmoid --method xavier

# Classificazione multi-classe (hidden sigmoid, output softmax)
./nnet-init.sh models/iris 4,16,3 --activation softmax --hidden-act sigmoid --method xavier

# Regressione (output lineare)
./nnet-init.sh models/sine 1,16,1 --activation linear --hidden-act tanh --method xavier
```

## nnet-run.sh

```bash
./nnet-run.sh <comando> <dataset> <model_dir> [opzioni]
```

**Comandi:** `train`, `predict`, `eval` (train + predict in sequenza)

### Opzioni di training

| Flag | Default | Descrizione |
|---|---|---|
| `--lr RATE` | 0.3 | Learning rate |
| `--epochs N` | 1000 | Numero massimo di epoche |
| `--optimizer OPT` | `sgd` | `sgd`, `sgd-momentum`, `sgd-momentum-decay`, `adam` |
| `--momentum M` | 0.0 | Coefficiente momentum |
| `--lr-decay D` | 0.0 | Decay del learning rate per epoca |
| `--loss FUNC` | `mse` | `mse`, `ce` (cross-entropy, solo con sigmoid/softmax) |
| `--task TASK` | `classification` | `classification` o `regression` |
| `--batch-size N` | 0 | Mini-batch size; 0 = full-batch gradient descent |
| `--val-split F` | 0 | Frazione del dataset per validazione (es. `0.2`) |
| `--patience N` | — | Early stopping: stop se val_mse non migliora per N epoche (richiede `--val-split`) |
| `--normalize` | — | Normalizzazione z-score degli input (stats salvate in `normalize.conf`) |
| `--dropout R` | 0 | Dropout rate sui layer hidden in training (inverted dropout) |
| `--clip N` | — | Gradient clipping element-wise |
| `--wd N` | 0 | L2 weight decay |
| `--no-save` | — | Non salvare i pesi dopo il training |
| `--use-best` | — | Carica il miglior checkpoint (`model_dir/best/`) invece dei pesi live |
| `--debug FLAG` | — | `forward`, `backward`, `update`, `network`, `metrics`, `all` |

### Preset optimizer

| `--optimizer` | lr | momentum | lr-decay |
|---|---|---|---|
| `sgd` | 0.3 | 0.0 | 0.0 |
| `sgd-momentum` | 0.5 | 0.9 | 0.0 |
| `sgd-momentum-decay` | 0.5 | 0.9 | 0.001 |
| `adam` | 0.001 | — | 0.0 |

I valori possono essere sovrascritti con i flag espliciti (`--lr`, `--momentum`, ecc.).

## Formato Dataset

```
# Commenti con #, righe vuote ignorate
# colonne: input1 input2 ... output1 output2 ...
0 0 0
0 1 1
1 0 1
1 1 0
```

Il bias viene aggiunto automaticamente — non includerlo nel dataset.

## Formato Modello

Ogni layer è un file `layer<N>.txt`:

```
ACTIVATION=sigmoid
0.512 -0.301  0.198   # pesi neurone 1 (ultimo = bias)
-0.104  0.432 -0.255  # pesi neurone 2
```

`model.conf` tiene i metadati dell'ultima sessione di training (optimizer, lr, best_mse, best_epoch, ecc.). `normalize.conf` viene creato da `--normalize` con mean/std per feature. Il checkpoint migliore è sempre in `model_dir/best/`.

## Esempi

### XOR (classificazione binaria)

```bash
./nnet-init.sh models/xor 2,8,1 --activation sigmoid --seed 42
./nnet-run.sh train dataset/xor.txt models/xor --epochs 3000 --lr 0.5
./nnet-run.sh predict dataset/xor.txt models/xor
```

### Iris (multi-classe, softmax + CCE)

```bash
./nnet-init.sh models/iris 4,16,3 --activation softmax --hidden-act sigmoid --seed 1
./nnet-run.sh train dataset/iris_toy.txt models/iris \
    --epochs 1000 --optimizer adam --loss ce
```

### Regressione seno

```bash
./nnet-init.sh models/sine 1,16,1 --activation linear --hidden-act tanh --method xavier
./nnet-run.sh train dataset/sine_regression.txt models/sine \
    --epochs 3000 --optimizer adam --task regression --normalize
```

### Mini-batch SGD con early stopping

```bash
./nnet-init.sh models/iris 4,16,3 --activation softmax --hidden-act sigmoid
./nnet-run.sh train dataset/iris_toy.txt models/iris \
    --epochs 2000 --optimizer adam --loss ce \
    --batch-size 8 --val-split 0.2 --patience 50
```

### Dropout + normalizzazione

```bash
./nnet-run.sh train dataset/iris_toy.txt models/iris \
    --epochs 1000 --optimizer adam \
    --normalize --dropout 0.2
```

## Test Suite

```bash
# Tutti i test (222 test, ~2 min)
bash tests/run.sh

# Solo una categoria
bash tests/run.sh batch
bash tests/run.sh dropout
bash tests/run.sh multiclass
```

Dispatcher disponibili: `unit`, `init`, `activations`, `optimizers`, `loss`, `checkpoint`, `pipeline`, `val_split`, `multiclass`, `metrics`, `normalize`, `dropout`, `batch`.

## Dipendenze

- `gawk` (GNU AWK)
- `bash` 4+
- `mkdir`, `ls` (Unix standard)

---

**Autore:** Luca Tagliavini
