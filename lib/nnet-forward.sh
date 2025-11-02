#!/bin/bash

# ===============================================================
# Forward pass multilayer
# ===============================================================

INPUT_FILE="$1"        # Dataset (es: dataset/xor.txt)
LAYERS_DIR="$2"        # Directory dei layer (es: models/xor)
TMPDIR="$3"            # Directory temporanea per gli output intermedi 
		       # In questa directory salviamo anche il file finale:
		       # File di output finale (es: tmp/xor-final.out)

# Va sempre aggiunto per tutti i layer:
ADD_BIAS=1             # Sempre aggiungere bias

mkdir -p "$TMPDIR"

# Copia iniziale: dataset diventa input iniziale
CURRENT_INPUT="$INPUT_FILE"

# Trova tutti i file dei layer, ordinati
LAYER_FILES=($(ls "$LAYERS_DIR"/layer*.txt | sort -V))

# Cicla sui layer
for ((i=0; i<${#LAYER_FILES[@]}; i++)); do
    LAYER_FILE="${LAYER_FILES[$i]}"
    LAYER_INDEX=$((i+1))

    # Output intermedio
    LAYER_OUTPUT="$TMPDIR/layer${LAYER_INDEX}.out"

    echo "[INFO] Forward layer ${LAYER_INDEX}: $LAYER_FILE" >&2

    # Esegui forward del layer
    awk \
        -v input_file="$CURRENT_INPUT" \
        -v layer_file="$LAYER_FILE" \
        -v output_file="$LAYER_OUTPUT" \
        -v add_bias="$ADD_BIAS" \
        -f lib/framework/utils-math.awk \
        -f lib/framework/utils-activation.awk \
        -f lib/framework/utils-functions.awk \
        -f lib/framework/nnet-forward-layer.awk /dev/null \
        /dev/null || exit 1

    # L'output di questo layer diventa input del prossimo
    CURRENT_INPUT="$LAYER_OUTPUT"
done

# Copia l'ultimo output nel file di output finale richiesto
OUTPUT_FILE=$LAYER_OUTPUT
echo "[INFO] Forward pass completato. Output: $OUTPUT_FILE" >&2
cat $OUTPUT_FILE
