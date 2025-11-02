#!/bin/bash

NET="$1"
MODE="$2"
THIRD="$3"
CONFIG="etc/${NET}.conf"
DATASET="dataset/${NET}.txt"
LAYERS_DIR="models/${NET}/"
TMPDIR="tmp/neural-${NET}"

[[ -z "$NET" || -z "$MODE" ]] && {
    echo "Usage:"
    echo "  $0 and train [epoche]"
    echo "  $0 xor guess \"1 0\""
    exit 1
}

[[ ! -f "$CONFIG" ]] && { echo "Config file not found: $CONFIG"; exit 1; }
[[ ! -d "$LAYERS_DIR" ]] && { echo "Layers not found: $LAYERS_DIR"; exit 1; }

source "$CONFIG"
mkdir -p "$TMPDIR"

if [[ "$MODE" == "train" ]]; then
    EPOCHS="${THIRD:-1000}"
    [[ ! -f "$DATASET" ]] && { echo "Dataset not found: $DATASET"; exit 1; }

    echo "[INFO] Training '$NET' for $EPOCHS epochs with $ACTIVATION_FUNC"
    bash lib/train.sh "$DATASET" "$LAYERS_DIR" "$TMPDIR" "$ACTIVATION_FUNC" "$LEARNING_RATE" "$EPOCHS"

elif [[ "$MODE" == "guess" ]]; then
    INPUT="$THIRD"
    [[ -z "$INPUT" ]] && { echo "Missing input for guess mode (e.g. \"1 0\")"; exit 1; }

    echo "$INPUT" > "$TMPDIR/custom_input.txt"
    echo "[INFO] Guessing output for: $INPUT with activation: $ACTIVATION_FUNC"
    bash lib/forward.sh "$TMPDIR/custom_input.txt" "$LAYERS_DIR" "$TMPDIR" "$ACTIVATION_FUNC"

else
    echo "[ERROR] Unknown mode: $MODE"
    exit 1
fi

