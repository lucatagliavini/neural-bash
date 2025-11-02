#!/bin/bash

DATASET="$1"       # es: dataset/xor.txt
LAYERS_DIR="$2"    # es: models/xor
TMPDIR="$3"        # es: tmp/xor-run
MAX_EPOCHS="${4:-1000}"
LEARNING_RATE="${5:-0.1}"
THRESHOLD="${6:-0.001}"

mkdir -p "$TMPDIR"

for ((epoch = 1; epoch <= MAX_EPOCHS; epoch++)); do
    #echo "# Epoch $epoch"

    bash lib/nnet-forward.sh "$DATASET" "$LAYERS_DIR" "$TMPDIR" 			>/dev/null 2>&1 || exit 1
    bash lib/nnet-backward.sh "$DATASET" "$LAYERS_DIR" "$TMPDIR" "$LEARNING_RATE" 	>/dev/null 2>&1 || exit 1
    bash lib/nnet-update.sh "$LAYERS_DIR" "$TMPDIR" "$LEARNING_RATE" 			>/dev/null 2>&1 || exit 1

    ERROR_LINE=$(awk \
        -v output_file="$TMPDIR/layer$(ls "$LAYERS_DIR" | grep -c 'layer').out" \
        -v target_file="$TMPDIR/target_only.txt" \
        -f lib/functions/nnet-error.awk $DATASET)

    #echo "$ERROR_LINE"
    ERROR_VALUE=$(echo "$ERROR_LINE" | cut -d= -f2)

    # Mostra errore ogni 100 epoche, o sempre alla prima o all’ultima
    if (( epoch == 1 || epoch % 100 == 0 || epoch == MAX_EPOCHS )); then
        printf "# Epoch %4d - %s\n" "$epoch" "$ERROR_LINE"
    fi

    # Convergenza?
    if [[ $(echo "$ERROR_VALUE < $THRESHOLD" | bc -l) -eq 1 ]]; then
        echo "# Converged (MSE=$ERROR_VALUE < $THRESHOLD) at epoch $epoch"
        break
    fi
done

