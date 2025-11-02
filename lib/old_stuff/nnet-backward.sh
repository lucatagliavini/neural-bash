#!/bin/bash

INPUT_DATASET="$1"     # es: dataset/xor.txt
LAYERS_DIR="$2"        # es: models/xor/
TMPDIR="$3"            # es: tmp/xor-run
LEARNING_RATE="${4:-0.1}"  # opzionale
DEBUG=1

mkdir -p "$TMPDIR"

# Estrai target (ultima colonna)
TARGET_FILE="$TMPDIR/target_only.txt"
awk '{ print $NF }' "$INPUT_DATASET" > "$TARGET_FILE"

# Conta quanti layer ci sono
LAYER_FILES=("$LAYERS_DIR"/layer*.txt)
NUM_LAYERS="${#LAYER_FILES[@]}"

for (( LAYER_INDEX=NUM_LAYERS; LAYER_INDEX>=1; LAYER_INDEX-- ))
do
    LAYER_FILE="$LAYERS_DIR/layer${LAYER_INDEX}.txt"
    ACTIVATION_FUNCTION=$(grep '^ACTIVATION=' "$LAYER_FILE" | cut -d= -f2)

    OUTPUT_FILE="$TMPDIR/layer${LAYER_INDEX}.out"
    DELTA_OUTPUT_FILE="$TMPDIR/layer${LAYER_INDEX}-delta.txt"
    GRADIENT_OUTPUT_FILE="$TMPDIR/layer${LAYER_INDEX}-grad.txt"

    if (( LAYER_INDEX == 1 )); then
        INPUT_FILE="$TMPDIR/input_only.txt"
    else
        PREV=$((LAYER_INDEX - 1))
        INPUT_FILE="$TMPDIR/layer${PREV}.out"
    fi

    # Per hidden layer, serve delta e pesi del layer successivo
    if (( LAYER_INDEX == NUM_LAYERS )); then
        # Output layer
        awk \
            -v output_file="$OUTPUT_FILE" \
            -v target_file="$TARGET_FILE" \
            -v input_file="$INPUT_FILE" \
            -v activation_function="$ACTIVATION_FUNCTION" \
            -v delta_output_file="$DELTA_OUTPUT_FILE" \
            -v gradient_output_file="$GRADIENT_OUTPUT_FILE" \
            -v learning_rate="$LEARNING_RATE" \
            -v debug=$DEBUG \
            -f lib/functions/math.awk \
            -f lib/functions/activation.awk \
            -f lib/functions/nnet-backward.awk \
            "$OUTPUT_FILE"
    else
        NEXT_LAYER=$((LAYER_INDEX + 1))
        NEXT_WEIGHTS_FILE="$LAYERS_DIR/layer${NEXT_LAYER}.txt"
        NEXT_DELTA_FILE="$TMPDIR/layer${NEXT_LAYER}-delta.txt"

        awk \
            -v output_file="$OUTPUT_FILE" \
            -v input_file="$INPUT_FILE" \
            -v activation_function="$ACTIVATION_FUNCTION" \
            -v delta_output_file="$DELTA_OUTPUT_FILE" \
            -v gradient_output_file="$GRADIENT_OUTPUT_FILE" \
            -v next_weights_file="$NEXT_WEIGHTS_FILE" \
            -v next_delta_file="$NEXT_DELTA_FILE" \
            -v learning_rate="$LEARNING_RATE" \
            -v debug=$DEBUG \
            -f lib/functions/math.awk \
            -f lib/functions/activation.awk \
            -f lib/functions/nnet-backward.awk \
            "$OUTPUT_FILE"
    fi
done

