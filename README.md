# Neural Network in AWK - Guida Completa

Sistema completo per il training e la predizione di reti neurali implementato in AWK.

**Versione**: 1.1-STABLE | [Changelog](CHANGELOG.md) | [Licenza MIT](LICENSE)

## 📁 Struttura del Progetto

```
.
├── lib/framework/
│   ├── utils-activation.awk    # Funzioni di attivazione (sigmoid, tanh, relu, leaky_relu)
│   ├── utils-shared.awk         # Funzioni condivise (matrici, logging, I/O)
│   ├── utils-network.awk        # Caricamento/salvataggio della rete
│   ├── utils-forward.awk        # Forward pass
│   ├── utils-backward.awk       # Backward pass (backpropagation)
│   ├── utils-update.awk         # Update dei pesi (gradient descent)
│   ├── utils-metrics.awk        # Metriche di valutazione (MSE, accuracy)
│   ├── nnet-train.awk           # Script principale per il training
│   └── nnet-predict.awk         # Script principale per la predizione
├── dataset/
│   ├── xor.txt                  # Dataset XOR
│   ├── and.txt                  # Dataset AND
│   └── or.txt                   # Dataset OR
├── models/
│   └── xor/                     # Modello XOR
│       ├── layer1.txt           # Pesi del primo layer
│       └── layer2.txt           # Pesi del secondo layer
├── nnet-train.sh                # Script wrapper per training
├── nnet-predict.sh              # Script wrapper per predizione
└── nnet-run.sh                  # Script unificato (train/predict/eval)
```

## 🚀 Quick Start

### 1. Training di Base

```bash
# Training con parametri di default
./nnet-run.sh train dataset/xor.txt models/xor

# Training con parametri personalizzati
./nnet-run.sh train dataset/xor.txt models/xor \
    --epochs 2000 \
    --lr 0.5 \
    --inputs 2 \
    --layers 2
```

### 2. Predizione

```bash
# Predizione con un modello già addestrato
./nnet-run.sh predict dataset/xor.txt models/xor
```

### 3. Training + Valutazione

```bash
# Training seguito immediatamente dalla valutazione
./nnet-run.sh eval dataset/xor.txt models/xor --epochs 1000
```

## 📋 Formato dei Dataset

I dataset devono essere file di testo con il seguente formato:

```
# Ogni riga rappresenta un campione
# Le prime N colonne sono gli input
# Le restanti colonne sono gli output attesi

# Esempio: XOR (2 input, 1 output)
0 0 0
0 1 1
1 0 1
1 1 0
```

**Nota importante:** Il bias viene aggiunto automaticamente dal sistema, non è necessario includerlo nel dataset.

## 🏗️ Formato dei Modelli

Ogni layer è salvato in un file separato (`layer1.txt`, `layer2.txt`, ecc.):

```
ACTIVATION=sigmoid
0.500000 -0.300000 0.200000
-0.100000 0.400000 -0.250000
```

- La prima riga specifica la funzione di attivazione
- Le righe successive contengono i pesi (una riga per neurone)
- Ogni colonna rappresenta un peso per un input (incluso il bias come ultima colonna)

## 🎯 Opzioni Disponibili

### Script Unificato (nnet-run.sh)

```bash
./nnet-run.sh <command> <dataset_file> <model_dir> [options]
```

**Comandi:**
- `train`: Addestra una rete neurale
- `predict`: Esegue predizioni con un modello addestrato
- `eval`: Addestra e poi valuta il modello

**Opzioni di Training:**
- `--inputs N`: Numero di feature in input (default: 2)
- `--layers N`: Numero di layer nella rete (default: 2)
- `--lr RATE`: Learning rate (default: 0.3)
- `--epochs N`: Numero massimo di epoche (default: 1000)
- `--no-save`: Non salvare il modello dopo il training
- `--debug FLAG`: Abilita output di debug

**Flag di Debug Disponibili:**
- `forward`: Debug del forward pass
- `backward`: Debug del backward pass
- `update`: Debug dell'aggiornamento pesi
- `network`: Debug del caricamento rete
- `metrics`: Debug delle metriche
- `all`: Tutti i debug abilitati

## 📊 Output del Training

Durante il training, viene mostrato l'MSE (Mean Squared Error):

```
[INFO] train: num_epochs = 1000
[EPOCH 1] MSE = 0.250000
[EPOCH 100] MSE = 0.125432
[EPOCH 200] MSE = 0.045678
...
[EPOCH 1000] MSE = 0.001234
[INFO] train: saving updated weights to models/xor
```

Se abilitato con `--print-result`, mostra anche le predizioni finali:

```
============================================================
FINAL PREDICTIONS
============================================================
[Sample 1] pred = 0.012345 | target = 0 | ✓
[Sample 2] pred = 0.987654 | target = 1 | ✓
[Sample 3] pred = 0.989012 | target = 1 | ✓
[Sample 4] pred = 0.023456 | target = 0 | ✓
============================================================
```

## 🔍 Output della Predizione

Il comando `predict` mostra una tabella dettagliata:

```
================================================================================
PREDICTIONS
================================================================================
Sample   | Predicted            | Target               | Status    
--------------------------------------------------------------------------------
1        | 0.012345             | 0                    | ✓ CORRECT
2        | 0.987654             | 1                    | ✓ CORRECT
3        | 0.989012             | 1                    | ✓ CORRECT
4        | 0.023456             | 0                    | ✓ CORRECT
================================================================================

EVALUATION METRICS
================================================================================
Mean Squared Error (MSE) : 0.001234
Accuracy                  : 100.00% (4/4)
================================================================================
```

## 🔧 Uso Avanzato

### Training con Debug Completo

```bash
./nnet-run.sh train dataset/xor.txt models/xor \
    --epochs 500 \
    --lr 0.3 \
    --debug all
```

### Uso Diretto degli Script AWK

Se preferisci usare AWK direttamente:

```bash
# Training
awk \
    -v dataset_file="dataset/xor.txt" \
    -v num_inputs=2 \
    -v model_dir="models/xor" \
    -v num_layers=2 \
    -v learning_rate=0.3 \
    -v max_epochs=1000 \
    -v save_model=1 \
    -v print_result=1 \
    -f lib/framework/utils-activation.awk \
    -f lib/framework/utils-shared.awk \
    -f lib/framework/utils-network.awk \
    -f lib/framework/utils-forward.awk \
    -f lib/framework/utils-backward.awk \
    -f lib/framework/utils-update.awk \
    -f lib/framework/utils-metrics.awk \
    -f lib/framework/nnet-train.awk \
    /dev/null

# Predizione
awk \
    -v dataset_file="dataset/xor.txt" \
    -v num_inputs=2 \
    -v model_dir="models/xor" \
    -v num_layers=2 \
    -f lib/framework/utils-activation.awk \
    -f lib/framework/utils-shared.awk \
    -f lib/framework/utils-network.awk \
    -f lib/framework/utils-forward.awk \
    -f lib/framework/utils-metrics.awk \
    -f lib/framework/nnet-predict.awk \
    /dev/null
```

## 🎓 Esempi Pratici

### Esempio 1: Training XOR

```bash
# Training del problema XOR classico
./nnet-run.sh train dataset/xor.txt models/xor \
    --epochs 2000 \
    --lr 0.5

# Verifica risultati
./nnet-run.sh predict dataset/xor.txt models/xor
```

### Esempio 2: Training AND con Valutazione

```bash
./nnet-run.sh eval dataset/and.txt models/and \
    --epochs 1000 \
    --lr 0.3
```

### Esempio 3: Training con Debug

```bash
# Debug solo del backward pass
./nnet-run.sh train dataset/or.txt models/or \
    --epochs 500 \
    --lr 0.4 \
    --debug backward
```

## 📝 Note Importanti

1. **Salvataggio Automatico**: Per impostazione predefinita, i pesi vengono salvati automaticamente dopo il training. Usa `--no-save` per disabilitare.

2. **Bias**: Il bias viene gestito automaticamente. Non includere una colonna bias nel dataset.

3. **Funzioni di Attivazione**: Attualmente supportate:
   - `sigmoid` (default) - **Raccomandato per output layer in classificazione binaria**
   - `tanh`
   - `relu` - **Solo per hidden layers** ⚠️ **MAI nell'output layer**
   - `leaky_relu` - **Hidden layers o output layer** (funziona ovunque)

   ⚠️ **IMPORTANTE**: Per problemi di classificazione binaria (output [0,1]) come XOR, AND, OR:
   - **Output Layer**: Usa `sigmoid` (consigliato), `tanh`, o `leaky_relu`
   - **Hidden Layers**: Puoi usare `relu` o `leaky_relu`
   - **MAI usare ReLU standard nell'output layer** (dying ReLU → nessun apprendimento)

   **Configurazioni consigliate per XOR**:
   ```bash
   # ✅ CONFIGURAZIONE OTTIMALE: ReLU hidden + Sigmoid output
   ./nnet-init.sh models/xor 2,8,1 --activation relu --activation-output sigmoid --method he
   ./nnet-run.sh train dataset/xor.txt models/xor --epochs 3000 --lr 0.5

   # ✅ ALTERNATIVA STABILE: Tutto sigmoid (più lento ma sempre converge)
   ./nnet-init.sh models/xor 2,4,1 --activation sigmoid --method xavier
   ./nnet-run.sh train dataset/xor.txt models/xor --epochs 5000 --lr 1.0

   # ❌ CONFIGURAZIONE ERRATA: Pochi neuroni + Xavier con ReLU
   ./nnet-init.sh models/xor 2,3,1 --activation relu --activation-output sigmoid --method xavier
   # Non converge: troppo pochi neuroni, inizializzazione sbagliata
   ```

   **Best Practices per ReLU**:
   - ✅ **Inizializzazione**: Usa **sempre He** (`--method he`) con ReLU, mai Xavier
   - ✅ **Neuroni nascosti**: Almeno **4-8 neuroni** per XOR (vs 2-3 per sigmoid)
   - ✅ **Learning rate**: Più alto (0.5-1.0) rispetto a sigmoid (0.3)
   - ✅ **Epoche**: Più epoche (3000-5000) per convergenza completa
   - ✅ **Output layer**: Usa **--activation-output sigmoid** per classificazione binaria

4. **Learning Rate**: Valori tipici tra 0.01 e 1.0. Sperimenta per trovare il valore ottimale.
   - **ReLU**: 0.5-1.0 (più alto)
   - **Sigmoid/Tanh**: 0.3-0.5 (moderato)

5. **Epoche**: Il training stampa l'MSE ogni 100 epoche per monitorare il progresso.
   - **ReLU**: 3000-5000 epoche per convergenza completa
   - **Sigmoid**: 1000-2000 epoche solitamente sufficienti

## 🐛 Troubleshooting

### Errore: File not found

Verifica che i path siano corretti e che i file esistano:
```bash
ls -la dataset/xor.txt
ls -la models/xor/
```

### MSE non converge con ReLU

**Sintomi**: Training con ReLU negli hidden layers converge molto lentamente o MSE rimane alto (>0.1).

**Cause comuni**:
1. **Inizializzazione sbagliata**: Stai usando Xavier invece di He
2. **Troppo pochi neuroni**: ReLU richiede più neuroni di sigmoid (4-8 vs 2-3 per XOR)
3. **Learning rate troppo basso**: ReLU funziona meglio con LR 0.5-1.0
4. **Poche epoche**: ReLU richiede più tempo per convergere (3000-5000 epoche)

**Soluzioni**:

```bash
# ❌ PROBLEMA: Configurazione non ottimale
./nnet-init.sh models/xor 2,3,1 --activation relu --activation-output sigmoid --method xavier
./nnet-run.sh train dataset/xor.txt models/xor --epochs 1000 --lr 0.3

# ✅ SOLUZIONE: Configurazione ottimale per ReLU
./nnet-init.sh models/xor 2,8,1 --activation relu --activation-output sigmoid --method he
./nnet-run.sh train dataset/xor.txt models/xor --epochs 3000 --lr 0.5

# ✅ ALTERNATIVA: Usa sigmoid se ReLU è problematico
./nnet-init.sh models/xor 2,4,1 --activation sigmoid --method xavier
./nnet-run.sh train dataset/xor.txt models/xor --epochs 2000 --lr 0.5
```

**Checklist per ReLU**:
- ✅ Usa `--method he` (non xavier)
- ✅ Almeno 4-8 neuroni nell'hidden layer
- ✅ Learning rate 0.5-1.0
- ✅ Almeno 3000 epoche
- ✅ Usa `--activation-output sigmoid` per output layer

### MSE non converge (generico)

Prova a:
- Aumentare il numero di epoche
- Modificare il learning rate (più alto o più basso)
- Verificare il formato del dataset
- Usare `--debug all` per identificare problemi

### Accuracy bassa

- Aumenta il numero di epoche
- Modifica l'architettura della rete (più neuroni/layer)
- Verifica il dataset per errori
- Controlla che stai usando l'inizializzazione corretta (He per ReLU, Xavier per sigmoid)

### MSE bloccato a 0.125-0.250 con predizioni costanti

**Causa**: Stai usando **ReLU standard** nell'output layer per classificazione binaria.

**Sintomi**:
```
[EPOCH 100] MSE = 0.250000 | LR = 0.300000 | LOSS(mse) = 0.250000
[EPOCH 200] MSE = 0.250000 | LR = 0.300000 | LOSS(mse) = 0.250000
...
Predicted: 0.000000, 0.000000, 0.000000, 0.000000  (o tutti 1.0)
```

**Problema**: "Dying ReLU" - quando preactivation < 0, la derivata è 0 → nessun gradiente → nessun apprendimento

**Soluzioni** (in ordine di preferenza):

```bash
# ✅ SOLUZIONE 1: Usa Sigmoid (raccomandato)
sed -i 's/ACTIVATION=relu/ACTIVATION=sigmoid/' models/xor/layer2.txt

# ✅ SOLUZIONE 2: Usa Leaky ReLU (funziona anche)
sed -i 's/ACTIVATION=relu/ACTIVATION=leaky_relu/' models/xor/layer2.txt

# ✅ SOLUZIONE 3: Reinizializza
./nnet-init.sh models/xor 2,4,1 --activation sigmoid --force
```

**Spiegazione tecnica**:
- **ReLU**: `f(x) = max(0, x)` → `f'(x) = 0` quando `x ≤ 0`
  - Output bloccato a 0 → derivata 0 → nessun apprendimento
- **Leaky ReLU**: `f(x) = max(0.01x, x)` → `f'(x) = 0.01` quando `x ≤ 0`
  - Piccolo gradiente anche quando negativo → continua ad apprendere
- **Sigmoid**: `f(x) = 1/(1+e^-x)` → output sempre in [0,1], gradiente mai 0

## 🐛 Bug Fix e Miglioramenti Recenti

### Versione 1.1 (Dicembre 2024)

**Correzioni Critiche**:

1. **Fix: Calcolo Delta Output Layer con ReLU** ([utils-activation.awk:60](lib/framework/utils-activation.awk#L60))
   - Problema: La funzione `compute_output_delta` usava sempre il valore post-attivazione per calcolare la derivata
   - Impatto: Backpropagation errata con ReLU/Leaky ReLU
   - Soluzione: Aggiunto parametro `preactivation` per usare il valore corretto:
     - ReLU/Leaky ReLU → pre-attivazione (z)
     - Sigmoid/Tanh → post-attivazione (y)

2. **Fix: MSE Uniformato** ([utils-loss.awk:39](lib/framework/utils-loss.awk#L39))
   - Problema: Incoerenza nel fattore 0.5 tra diverse funzioni MSE
   - Soluzione: Uniformato uso di `0.5 * (y - t)²` per semplificare la derivata

3. **Fix: Backward Pass per Output Layer** ([utils-backward.awk:47](lib/framework/utils-backward.awk#L47))
   - Problema: Pre-attivazione non passata al calcolo del delta dell'output layer
   - Soluzione: Aggiunta estrazione e passaggio di `preactivation`

4. **Fix: Stampa Predizioni durante Training** ([utils-forward.awk:96](lib/framework/utils-forward.awk#L96))
   - Problema: Accesso errato all'array `output_layer_info` con indice doppio
   - Soluzione: Corretto accesso con chiave semplice `output_layer_info["num_neurons"]`

**Validazione**:
- ✅ Training con ReLU funzionante
- ✅ Training con Sigmoid invariato
- ✅ MSE converge correttamente
- ✅ Predizioni stampate correttamente

## 🔮 Funzionalità Future

Possibili estensioni del sistema:

1. ~~**Inizializzazione Pesi**: Implementare Xavier/He initialization~~ ✅ **IMPLEMENTATO**
2. ~~**Ottimizzatori**: Adam, RMSprop, momentum~~ ✅ **IMPLEMENTATO**
3. **Regolarizzazione**: L1/L2, dropout
4. **Batch Processing**: Mini-batch gradient descent
5. **Early Stopping**: Fermare il training automaticamente
6. **Cross-Validation**: K-fold validation
7. **Export/Import**: Formati standard (JSON, CSV)

## 📚 Risorse

- Documentazione AWK: https://www.gnu.org/software/gawk/manual/
- Neural Networks: https://www.deeplearningbook.org/
- Backpropagation: http://neuralnetworksanddeeplearning.com/

## 🤝 Contributi

Suggerimenti per miglioramenti:

1. Creare test automatizzati
2. Aggiungere più funzioni di loss
3. Implementare data augmentation
4. Aggiungere visualizzazione dei risultati
5. Supporto per dataset più grandi

---

**Autore**: Luca Tagliavini
**Versione**: 1.1-STABLE
**Licenza**: MIT (vedi [LICENSE](LICENSE))
**Ultimo Aggiornamento**: Dicembre 2024
