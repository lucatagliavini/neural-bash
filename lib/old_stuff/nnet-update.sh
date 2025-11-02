#!/bin/bash

LAYERS_DIR="$1"            # es: models/xor/
TMPDIR="$2"                # es: tmp/xor-run
LEARNING_RATE="${3:-0.1}"  # default 0.1

# Conta quanti layer ci sono
LAYER_FILES=("$LAYERS_DIR"/layer*.txt)
NUM_LAYERS="${#LAYER_FILES[@]}"

for LAYER_FILE in "${LAYER_FILES[@]}"; do
    BASENAME=$(basename "$LAYER_FILE")
    LAYER_INDEX=$(echo "$BASENAME" | sed -E 's/layer([0-9]+)\.txt/\1/')

    echo "# Updating layer $LAYER_INDEX..." >&2

    GRAD_FILE="$TMPDIR/layer${LAYER_INDEX}-grad.txt"
    TMP_OUTPUT_FILE="$TMPDIR/layer${LAYER_INDEX}-updated.txt"

    # Scegli input corretto:
    if (( LAYER_INDEX == 1 )); then
        INPUT_FILE="$TMPDIR/input_only.txt"
    else
        PREV=$((LAYER_INDEX - 1))
        INPUT_FILE="$TMPDIR/layer${PREV}_with_bias.tmp"
    fi

    awk \
        -v input_file="$INPUT_FILE" \
        -v grad_file="$GRAD_FILE" \
        -v weights_file="$LAYER_FILE" \
        -v output_file="$TMP_OUTPUT_FILE" \
        -v learning_rate="$LEARNING_RATE" \
        -v debug=0 \
        -f lib/functions/math.awk \
        -f lib/functions/shared-functions.awk \
        -f lib/functions/nnet-update.awk \
        $INPUT_FILE

    mv "$TMP_OUTPUT_FILE" "$LAYER_FILE"
done

