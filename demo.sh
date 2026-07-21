#!/bin/bash
#
# Script di esempio per dimostrare l'uso del sistema di Neural Network in AWK
#
# Questo script esegue una serie di esperimenti con diversi dataset
# per mostrare le capacità del sistema.
#

set -e

echo "=========================================="
echo "Neural Network in AWK - Demo Script"
echo "=========================================="
echo ""

# Verifica che gli script siano eseguibili
chmod +x nnet-run.sh nnet-train.sh nnet-predict.sh

# ============================================================================
# ESPERIMENTO 1: XOR Problem
# ============================================================================

echo "=========================================="
echo "ESPERIMENTO 1: XOR Problem"
echo "=========================================="
echo ""
echo "Il problema XOR è un classico esempio di problema non linearmente separabile."
echo "Richiede almeno un hidden layer per essere risolto correttamente."
echo ""
read -p "Premi ENTER per iniziare il training XOR..."
echo ""

./nnet-run.sh train dataset/xor.txt models/xor \
    --epochs 2000 \
    --lr 0.5 \
    --inputs 2 \
    --layers 2

echo ""
read -p "Premi ENTER per vedere le predizioni..."
echo ""

./nnet-run.sh predict dataset/xor.txt models/xor \
    --inputs 2 \
    --layers 2

echo ""
echo "✓ XOR completato!"
echo ""

# ============================================================================
# ESPERIMENTO 2: AND Logic Gate
# ============================================================================

echo "=========================================="
echo "ESPERIMENTO 2: AND Logic Gate"
echo "=========================================="
echo ""
echo "La funzione AND è linearmente separabile, quindi dovrebbe convergere rapidamente."
echo ""
read -p "Premi ENTER per iniziare il training AND..."
echo ""

./nnet-run.sh eval dataset/and.txt models/and \
    --epochs 1000 \
    --lr 0.3 \
    --inputs 2 \
    --layers 1

echo ""
echo "✓ AND completato!"
echo ""

# ============================================================================
# ESPERIMENTO 3: OR Logic Gate
# ============================================================================

echo "=========================================="
echo "ESPERIMENTO 3: OR Logic Gate"
echo "=========================================="
echo ""
echo "La funzione OR è anch'essa linearmente separabile."
echo ""
read -p "Premi ENTER per iniziare il training OR..."
echo ""

./nnet-run.sh eval dataset/or.txt models/or \
    --epochs 1000 \
    --lr 0.3 \
    --inputs 2 \
    --layers 1

echo ""
echo "✓ OR completato!"
echo ""

# ============================================================================
# ESPERIMENTO 4: Confronto Learning Rates
# ============================================================================

echo "=========================================="
echo "ESPERIMENTO 4: Confronto Learning Rates su XOR"
echo "=========================================="
echo ""
echo "Testiamo diversi learning rate per vedere l'impatto sulla convergenza."
echo ""

learning_rates=(0.1 0.3 0.5 0.8 1.0)

for lr in "${learning_rates[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Testing Learning Rate: $lr"
    echo "----------------------------------------"
    
    # Training con questo learning rate
    ./nnet-run.sh train dataset/xor.txt models/xor \
        --epochs 1000 \
        --lr $lr \
        --inputs 2 \
        --layers 2 | grep "EPOCH"
    
    echo ""
done

echo ""
echo "✓ Confronto learning rates completato!"
echo ""

# ============================================================================
# ESPERIMENTO 5: Debug Mode
# ============================================================================

echo "=========================================="
echo "ESPERIMENTO 5: Debug Mode"
echo "=========================================="
echo ""
echo "Esempio di training con output di debug per il backward pass."
echo "Questo è utile per capire cosa succede durante il training."
echo ""
read -p "Premi ENTER per vedere il debug del backward pass (primi 10 epoch)..."
echo ""

./nnet-run.sh train dataset/xor.txt models/xor \
    --epochs 10 \
    --lr 0.5 \
    --inputs 2 \
    --layers 2 \
    --debug backward | head -n 50

echo ""
echo "✓ Demo debug completata!"
echo ""

# ============================================================================
# RIEPILOGO
# ============================================================================

echo "=========================================="
echo "RIEPILOGO ESPERIMENTI"
echo "=========================================="
echo ""
echo "Hai completato tutti gli esperimenti dimostrativi!"
echo ""
echo "Modelli addestrati disponibili in:"
echo "  - models/xor/"
echo "  - models/and/"
echo "  - models/or/"
echo ""
echo "Puoi ora:"
echo ""
echo "1. Testare i modelli con nuovi dati:"
echo "   ./nnet-run.sh predict dataset/xor.txt models/xor"
echo ""
echo "2. Continuare il training con più epoche:"
echo "   ./nnet-run.sh train dataset/xor.txt models/xor --epochs 5000"
echo ""
echo "3. Sperimentare con diversi parametri:"
echo "   ./nnet-run.sh train dataset/xor.txt models/xor --lr 0.7 --epochs 3000"
echo ""
echo "4. Creare i tuoi dataset e modelli personalizzati!"
echo ""
echo "Per maggiori informazioni, consulta il README.md"
echo ""
echo "=========================================="
echo "Demo completata! Buon machine learning!"
echo "=========================================="
