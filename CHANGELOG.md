# Changelog

Tutte le modifiche significative al progetto sono documentate in questo file.

## [1.1-STABLE] - 2024-12-06

### ✨ Nuove Funzionalità

#### Supporto per Architetture Miste
**File modificati**: `nnet-init.sh`, `lib/framework/nnet-init.awk`

- **Feature**: Aggiunto parametro `--activation-output` per specificare una funzione di attivazione diversa per l'output layer
- **Uso**: `./nnet-init.sh models/xor 2,8,1 --activation relu --activation-output sigmoid --method he`
- **Vantaggi**:
  - Non serve più modificare manualmente i file layer
  - Facilita creazione di architetture miste (ReLU hidden + Sigmoid output)
  - Best practice integrata per classificazione binaria
- **Esempio pratico**: ReLU negli hidden layers (veloce) + Sigmoid nell'output (stabile per [0,1])

### 🐛 Bug Fix Critici

#### Correzione Backpropagation con ReLU
**File modificati**: `lib/framework/utils-activation.awk`, `lib/framework/utils-backward.awk`

- **Problema**: La funzione `compute_output_delta` usava sempre il valore post-attivazione per calcolare la derivata, causando backpropagation errata con ReLU/Leaky ReLU
- **Soluzione**: Aggiunto parametro `preactivation` per usare il valore corretto basato sulla funzione di attivazione:
  - ReLU/Leaky ReLU → usa pre-attivazione (z)
  - Sigmoid/Tanh → usa post-attivazione (y)
- **Impatto**: Training con ReLU ora funziona correttamente negli hidden layers

#### MSE Uniformato
**File modificato**: `lib/framework/utils-loss.awk`

- **Problema**: Incoerenza nel fattore 0.5 tra diverse funzioni MSE
- **Soluzione**: Uniformato uso di `0.5 * (y - t)²` in tutte le funzioni per semplificare la derivata durante backpropagation
- **Impatto**: Calcolo della loss più consistente e matematicamente corretto

#### Stampa Predizioni durante Training
**File modificato**: `lib/framework/utils-forward.awk`

- **Problema**: Accesso errato all'array `output_layer_info` con indice doppio, causando stampa vuota delle predizioni
- **Soluzione**: Corretto accesso con chiave semplice `output_layer_info["num_neurons"]`
- **Impatto**: Le predizioni durante il training ora vengono stampate correttamente

### 📚 Documentazione Migliorata

#### README.md
- Aggiunta sezione "Bug Fix e Miglioramenti Recenti" con dettagli completi
- Aggiunta guida sulla scelta corretta delle funzioni di attivazione
- Nuovo troubleshooting: "MSE bloccato a 0.125 con predizioni ~0.5"
- Nuovo troubleshooting: "MSE non converge con ReLU" con soluzioni dettagliate
- **Best Practices per ReLU**: Configurazioni consigliate con esempi pratici
- Esempi di architetture corrette e sbagliate (ottimale vs errata)
- Avvisi su ReLU nell'output layer per classificazione binaria
- Tabelle comparative: Learning rate e epoche per ReLU vs Sigmoid

#### README-INIT.md
- Aggiunta tabella con opzione `--force`
- Avviso importante sulla scelta della funzione di attivazione
- Nuovo esempio 4: "Architettura Mista" per classificazione binaria
- **Sezione Best Practices per ReLU**: Guida completa su quando e come usare ReLU
- Configurazioni consigliate per XOR con comparazione ottimale vs errata
- Checklist estesa con parametri specifici per ReLU e Sigmoid
- Regole generali per ReLU vs Sigmoid/Tanh
- Note sulla versione 1.1 con link al changelog principale

#### .gitignore
- Creato file `.gitignore` per escludere modelli addestrati e file temporanei
- Mantiene i modelli demo per documentazione
- Esclude file OS, IDE e backup

### ✅ Validazione

Tutti i bug fix sono stati testati e validati:
- ✅ Training con ReLU funzionante negli hidden layers
- ✅ Training con Sigmoid invariato e corretto
- ✅ MSE converge correttamente (testato su XOR)
- ✅ Predizioni stampate correttamente durante training
- ✅ Accuracy 100% su dataset XOR con configurazione corretta

### ⚠️ Note Importanti

**Breaking Changes**: Nessuno - tutte le modifiche sono retrocompatibili

**Raccomandazioni**:
- Per problemi di classificazione binaria (XOR, AND, OR), usa **sempre sigmoid nell'output layer**
- ReLU va bene per hidden layers, ma NON per output layer in classificazione [0,1]
- **ReLU richiede configurazione specifica**: He initialization, più neuroni (4-8 vs 2-3), LR alto (0.5-1.0), più epoche (3000-5000)
- **Configurazione ottimale per XOR**: `./nnet-init.sh models/xor 2,8,1 --activation relu --activation-output sigmoid --method he`
- Se hai modelli addestrati con versione < 1.1, considera di ri-addestrali per risultati ottimali

### 🔗 Collegamenti

- [README principale](README.md)
- [Guida inizializzazione](README-INIT.md)
- [Licenza MIT](LICENSE)

---

## [1.0-ALFA] - 2024-11 (Pre-release)

### Funzionalità Iniziali

- Implementazione base rete neurale in AWK
- Supporto per funzioni di attivazione: sigmoid, tanh, relu, leaky_relu
- Inizializzazione pesi: Xavier, He, Random
- Optimizer: SGD, Adam, Momentum
- Forward e backward propagation
- Training e predizione
- Dataset: XOR, AND, OR
- Script wrapper bash per facilità d'uso

### File Principali

- Framework AWK modulare in `lib/framework/`
- Script bash `nnet-run.sh`, `nnet-init.sh`
- Documentazione README.md e README-INIT.md
