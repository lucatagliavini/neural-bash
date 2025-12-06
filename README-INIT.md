# 🎲 nnet-init.sh - Inizializzazione Pesi Neural Network

## Descrizione

Script per inizializzare i pesi di una neural network con diverse strategie di inizializzazione (Xavier, He, Random).

## 🚀 Quick Start

```bash
# Inizializza modello XOR (2 inputs, 3 hidden, 1 output)
./nnet-init.sh models/xor 2,3,1

# Inizializza con ReLU e He initialization
./nnet-init.sh models/custom 4,8,8,2 --activation relu --method he

# Con seed riproducibile
./nnet-init.sh models/test 2,4,1 --seed 42
```

## 📋 Sintassi

```bash
./nnet-init.sh <model_dir> <architecture> [options]
```

### Argomenti Posizionali

| Argomento | Descrizione | Esempio |
|-----------|-------------|---------|
| `model_dir` | Directory dove salvare il modello | `models/xor` |
| `architecture` | Architettura rete (layer separati da virgola) | `2,3,1` |

**Formato architettura:**
```
input_size,hidden1_size,hidden2_size,...,output_size
```

### Opzioni

| Opzione | Default | Valori | Descrizione |
|---------|---------|--------|-------------|
| `--activation FUNC` | `sigmoid` | `sigmoid`, `tanh`, `relu`, `leaky_relu` | Funzione di attivazione |
| `--method METHOD` | `xavier` | `xavier`, `he`, `random` | Metodo inizializzazione pesi |
| `--seed N` | (random) | Numero intero | Seed per riproducibilità |
| `-h`, `--help` | - | - | Mostra help |

## 📚 Esempi Pratici

### Esempio 1: XOR Problem
```bash
# Architettura classica per XOR
./nnet-init.sh models/xor 2,3,1

# Output:
# ==========================================
# NEURAL NETWORK INITIALIZATION
# ==========================================
# Model directory  : models/xor
# Architecture     : 2,3,1
# Activation       : sigmoid
# Init method      : xavier
# ==========================================
# 
# [INFO] Creating layer1.txt (3 neurons, 3 inputs including bias)
# [INFO] Creating layer2.txt (1 neurons, 4 inputs including bias)
# 
# ==========================================
# INITIALIZATION COMPLETED!
# ==========================================
# 
# Model structure:
#   Input layer:  2 neurons
#   Hidden layer 1: 3 neurons (9 weights)
#   Output layer: 1 neurons (4 weights)
# 
# Total layers: 2
# Total weights: 13
```

### Esempio 2: Deep Network con ReLU
```bash
# Network profonda per problemi complessi
./nnet-init.sh models/deep 10,20,20,10,5,2 --activation relu --method he

# Struttura:
# - Input: 10 features
# - Hidden 1: 20 neurons
# - Hidden 2: 20 neurons
# - Hidden 3: 10 neurons
# - Hidden 4: 5 neurons
# - Output: 2 classes
```

### Esempio 3: Inizializzazione Riproducibile
```bash
# Con seed fisso per debugging
./nnet-init.sh models/debug 2,4,1 --seed 42

# Ogni esecuzione produrrà gli stessi pesi
```

### Esempio 4: Multi-Class Classification
```bash
# Classificazione 4 classi
./nnet-init.sh models/multiclass 8,16,8,4 \
    --activation relu \
    --method he \
    --seed 123
```

## 🎯 Metodi di Inizializzazione

### Xavier Initialization (Default)
**Quando usarlo:** Con sigmoid e tanh

**Formula:** Uniform distribution in `[-limit, limit]` dove `limit = sqrt(6/(fan_in + fan_out))`

**Vantaggi:**
- Mantiene varianza simile tra layer
- Previene vanishing/exploding gradients
- Ottimale per sigmoid/tanh

```bash
./nnet-init.sh models/xor 2,3,1 --method xavier
```

### He Initialization
**Quando usarlo:** Con ReLU e varianti

**Formula:** Normal distribution `N(0, sqrt(2/fan_in))`

**Vantaggi:**
- Progettato specificamente per ReLU
- Compensa per neuroni "morti"
- Migliore convergenza con ReLU

```bash
./nnet-init.sh models/relu 4,8,2 --activation relu --method he
```

### Random Initialization
**Quando usarlo:** Per testing/debugging

**Formula:** Uniform distribution in `[-0.5, 0.5]`

**Nota:** Non raccomandato per training serio

```bash
./nnet-init.sh models/test 2,4,1 --method random
```

## 📊 Architetture Comuni

### Logic Gates (AND, OR, XOR)
```bash
# Input: 2, Hidden: 3, Output: 1
./nnet-init.sh models/logic 2,3,1
```

### Binary Classification
```bash
# 4 features → 8 hidden → 1 output (0/1)
./nnet-init.sh models/binary 4,8,1 --activation sigmoid
```

### Multi-Class Classification
```bash
# 10 features → 20 hidden → 5 classes
./nnet-init.sh models/multiclass 10,20,5 --activation relu --method he
```

### Deep Network
```bash
# Network profonda con 4 hidden layers
./nnet-init.sh models/deep 8,16,16,8,8,3 --activation relu --method he
```

### Autoencoder
```bash
# Encoder-Decoder simmetrico
./nnet-init.sh models/autoencoder 10,8,4,8,10 --activation relu --method he
```

## 🔧 Workflow Completo

### 1. Inizializza Modello
```bash
./nnet-init.sh models/mymodel 4,8,8,2 --activation relu --method he
```

### 2. Verifica File Creati
```bash
ls -la models/mymodel/
# Output:
# layer1.txt  (8 neurons × 5 inputs = 40 weights)
# layer2.txt  (8 neurons × 9 inputs = 72 weights)
# layer3.txt  (2 neurons × 9 inputs = 18 weights)
```

### 3. Visualizza un Layer
```bash
cat models/mymodel/layer1.txt
# Output:
# ACTIVATION=relu
# 0.234567 -0.123456 0.345678 0.456789 -0.234567
# -0.345678 0.456789 -0.567890 0.678901 0.123456
# ...
```

### 4. Train il Modello
```bash
./nnet-run.sh train dataset/mydata.txt models/mymodel \
    --inputs 4 \
    --layers 3 \
    --epochs 2000 \
    --lr 0.3
```

### 5. Test il Modello
```bash
./nnet-run.sh predict dataset/mydata.txt models/mymodel \
    --inputs 4 \
    --layers 3
```

## 🐛 Troubleshooting

### Problema: Directory già esistente
```bash
# Soluzione 1: Forza sovrascrittura (automaticamente chiede conferma)
./nnet-init.sh models/existing 2,3,1

# Soluzione 2: Usa nome diverso
./nnet-init.sh models/existing_v2 2,3,1

# Soluzione 3: Rimuovi directory esistente
rm -rf models/existing
./nnet-init.sh models/existing 2,3,1
```

### Problema: Formato architettura errato
```bash
# ERRATO
./nnet-init.sh models/test 2-3-1        # usa virgole, non trattini
./nnet-init.sh models/test 2 3 1        # usa virgole, non spazi
./nnet-init.sh models/test 2,1          # almeno 2 layer (input e output)

# CORRETTO
./nnet-init.sh models/test 2,3,1
```

### Problema: Activation function non valida
```bash
# ERRATO
./nnet-init.sh models/test 2,3,1 --activation softmax  # non supportato

# CORRETTO (usa una funzione supportata)
./nnet-init.sh models/test 2,3,1 --activation sigmoid
./nnet-init.sh models/test 2,3,1 --activation relu
```

## 📝 Note Importanti

### 1. Bias Automatico
I bias sono automaticamente inclusi! Non serve specificarli nell'architettura.

```bash
# Architettura: 2,3,1
# Significa:
# - Layer 1: 3 neuroni, ognuno ha 3 pesi (2 input + 1 bias)
# - Layer 2: 1 neurone, ha 4 pesi (3 input + 1 bias)
```

### 2. Numero di Layer
Il numero di layer si riferisce ai layer con pesi (non include l'input layer).

```bash
# 2,3,1 = 2 layer (1 hidden + 1 output)
# 2,4,4,1 = 3 layer (2 hidden + 1 output)
```

### 3. Compatibilità Training
Il parametro `--layers` nel training deve corrispondere al numero di layer file:

```bash
# Inizializzazione con 2,4,4,1 crea 3 layer file
./nnet-init.sh models/test 2,4,4,1

# Training deve usare --layers 3
./nnet-run.sh train dataset.txt models/test --inputs 2 --layers 3
```

### 4. Funzioni di Attivazione
La funzione di attivazione viene salvata in ogni layer file e usata automaticamente durante il training.

### 5. Riproducibilità
Usa `--seed` per ottenere sempre gli stessi pesi iniziali:

```bash
# Questi due comandi producono identici layer file
./nnet-init.sh models/test1 2,3,1 --seed 42
./nnet-init.sh models/test2 2,3,1 --seed 42
```

## 🔬 Alternativa: Script AWK

Esiste anche una versione AWK più leggera:

```bash
# Uso diretto
awk -f nnet-init.awk \
    -v model_dir="models/xor" \
    -v architecture="2,3,1" \
    -v activation="sigmoid" \
    -v init_method="xavier" \
    -v seed=42 \
    /dev/null

# Oppure renderlo eseguibile
chmod +x nnet-init.awk
./nnet-init.awk \
    -v model_dir="models/xor" \
    -v architecture="2,3,1" \
    /dev/null
```

**Vantaggi versione AWK:**
- ✅ Più veloce (tutto in un processo)
- ✅ Più portabile (solo AWK richiesto)
- ✅ Matematica più precisa

**Vantaggi versione Bash:**
- ✅ Più leggibile
- ✅ Validazione più robusta
- ✅ Messaggi di errore migliori

## 📚 Riferimenti

- **Xavier Initialization:** "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)
- **He Initialization:** "Delving Deep into Rectifiers" (He et al., 2015)

## ✅ Checklist

Prima di iniziare il training:

- [ ] Modello inizializzato con architettura corretta
- [ ] Funzione di attivazione appropriata (sigmoid per sigmoid, he per relu)
- [ ] Metodo di inizializzazione corretto
- [ ] File layer verificati (controllare con `cat models/*/layer*.txt`)
- [ ] Dataset preparato con formato corretto
- [ ] Parametri training corrispondenti (`--inputs` e `--layers`)

---

## 🐛 Note sulla Versione 1.1

La versione 1.1 include importanti correzioni al backpropagation che risolvono problemi con ReLU e altre funzioni di attivazione. Se hai modelli addestrati con versioni precedenti (< 1.1), ti consigliamo di ri-addestrali per ottenere risultati ottimali.

**Modifiche principali**:
- ✅ Corretto calcolo del delta per output layer con ReLU
- ✅ MSE uniformato con fattore 0.5
- ✅ Stampa predizioni durante training funzionante

Per maggiori dettagli, vedi il [README principale](README.md#-bug-fix-e-miglioramenti-recenti).

---

**Pronto per inizializzare la tua neural network! 🎲**
