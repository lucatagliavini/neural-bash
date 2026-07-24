# nnet-init.sh — Inizializzazione Pesi Neural Network

Script per creare una nuova rete neurale con pesi inizializzati secondo diverse strategie.

## Sintassi

```bash
./nnet-init.sh <model_dir> <architettura> [opzioni]
```

```bash
# Esempi rapidi
./nnet-init.sh models/xor 2,3,1
./nnet-init.sh models/iris 4,16,3 --activation softmax --hidden-act sigmoid --method xavier
./nnet-init.sh models/sine 1,16,1 --activation linear --hidden-act tanh --seed 42
```

## Argomenti

| Argomento | Descrizione | Esempio |
|---|---|---|
| `model_dir` | Directory dove salvare il modello | `models/xor` |
| `architettura` | Layer separati da virgola: `input,hidden...,output` | `2,8,1` o `4,16,8,3` |

## Opzioni

| Opzione | Default | Descrizione |
|---|---|---|
| `--activation FUNC` | `sigmoid` | Attivazione per tutti i layer (hidden e output) |
| `--hidden-act FUNC` | (uguale a `--activation`) | Attivazione solo per i layer hidden; sovrascrive `--activation` per i layer intermedi |
| `--method METHOD` | `xavier` | Strategia di inizializzazione dei pesi |
| `--seed N` | (casuale) | Seed per risultati riproducibili |
| `--force` | — | Sovrascrive una directory esistente senza chiedere conferma |
| `-h`, `--help` | — | Mostra help |

## Attivazioni disponibili

| Funzione | Uso consigliato |
|---|---|
| `sigmoid` | Classificazione binaria (output), hidden layer generici |
| `tanh` | Hidden layer, convergenza spesso più veloce di sigmoid |
| `relu` | Hidden layer profondi, usa `--method he` |
| `leaky_relu[:alpha]` | Come relu ma evita neuroni morti; alpha default 0.01 |
| `softmax` | **Solo output layer** per classificazione multi-classe |
| `linear` | **Solo output layer** per regressione |

## Metodi di inizializzazione

### Xavier (default)
Distribuisce i pesi in `[-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))]`.
Mantiene la varianza stabile tra layer. **Consigliato con sigmoid e tanh.**

### He
Campiona da `N(0, √(2/fan_in))`.
Compensa la soppressione dei valori negativi di ReLU. **Consigliato con relu e leaky_relu.**

### Random
Distribuzione uniforme in `[-0.5, 0.5]`. Utile per test e debugging.

## Esempi

### Classificazione binaria (XOR, AND, OR)
```bash
./nnet-init.sh models/xor 2,8,1 --activation sigmoid --method xavier --seed 42
```

### Classificazione multi-classe (softmax output)
```bash
# Hidden layer con sigmoid, output con softmax
./nnet-init.sh models/iris 4,16,3 --activation softmax --hidden-act sigmoid
```

### Regressione (output lineare)
```bash
# Hidden layer con tanh, output lineare (nessuna attivazione)
./nnet-init.sh models/sine 1,16,1 --activation linear --hidden-act tanh --method xavier
```

### Rete profonda con ReLU
```bash
./nnet-init.sh models/deep 10,64,32,16,5 --activation relu --method he
```

### Leaky ReLU con alpha personalizzato
```bash
./nnet-init.sh models/lrelu 2,8,1 --activation leaky_relu:0.1
```

## File generati

Dopo l'inizializzazione troverai in `model_dir/`:

```
layer1.txt    # Pesi del primo layer (hidden o output se rete a 1 layer)
layer2.txt    # Pesi del secondo layer
...
model.conf    # Metadati: architettura, activation, init_method
```

Ogni `layer<N>.txt` ha questo formato:
```
ACTIVATION=sigmoid
w1 w2 ... bias    # una riga per neurone; l'ultimo peso è sempre il bias
w1 w2 ... bias
```

Il numero di colonne per riga è `num_inputs_del_layer + 1` (il +1 è il bias).

## Note

**Bias automatico** — non includere una colonna bias nel dataset: viene aggiunto internamente.

**Numero di layer** — `2,8,4,1` genera 3 file layer (hidden1, hidden2, output). Il valore `--layers` in `nnet-run.sh` viene auto-rilevato dal numero di file `layer*.txt`.

**Riproducibilità** — con lo stesso `--seed` e la stessa architettura si ottengono sempre gli stessi pesi iniziali, indipendentemente dalla piattaforma.

**`--force`** — senza questo flag, se `model_dir` esiste già viene chiesta conferma prima di sovrascrivere.
