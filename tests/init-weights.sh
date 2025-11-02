#!/bin/bash

MODEL="$1"           # es: xor
LAYERS_DIR="models/${MODEL}"
CONFIG_FILE="etc/${MODEL}.conf"
INIT_RANGE="${2:-0.5}"  # default: 0.5

[[ -z "$MODEL" || ! -f "$CONFIG_FILE" ]] && {
    echo "Usage: $0 model_name [init_range]"
    echo "Example: $0 xor 0.3"
    exit 1
}

mkdir -p "$LAYERS_DIR"

# Legge la struttura della rete dal file di configurazione
# es: LAYER_DIMS="2 2 1"
source "$CONFIG_FILE"

[[ -z "$LAYER_DIMS" ]] && {
    echo "Missing LAYER_DIMS in $CONFIG_FILE"
    exit 1
}

read -ra DIMS <<< "$LAYER_DIMS"
NUM_LAYERS=$(( ${#DIMS[@]} - 1 ))

echo "[INFO] Inizializzo $NUM_LAYERS layer con range ±$INIT_RANGE"

for (( L=1; L<=NUM_LAYERS; L++ ))
do
    INPUT_DIM=${DIMS[$((L-1))]}
    OUTPUT_DIM=${DIMS[$L]}
    LAYER_FILE="$LAYERS_DIR/layer${L}.txt"
    echo "[INFO] → Layer $L: $INPUT_DIM → $OUTPUT_DIM"

    : > "$LAYER_FILE"
    for (( n=1; n<=OUTPUT_DIM; n++ ))
    do
        LINE=""
        for (( i=0; i<=INPUT_DIM; i++ ))  # +1 per il bias
        do
            R=$(awk -v min=-1 -v max=1 'BEGIN { printf "%.6f", min + rand() * (max - min) }')
            V=$(awk -v r="$R" -v s="$INIT_RANGE" 'BEGIN { printf "%.6f", r * s }')
            LINE+="$V "
        done
        echo "${LINE::-1}" >> "$LAYER_FILE"
    done
done
