#!/bin/bash

INPUT_DATASET="$1"       # es: dataset/xor.txt
LAYERS_DIR="$2"          # es: models/xor/
TMPDIR="$3"              # es: tmp/xor-run

mkdir -p "$TMPDIR"
INPUT_FILE="$TMPDIR/input_only.txt"

# Estrai input rimuovendo l'ultima colonna e aggiungendo bias=1
awk '{
    NF--;                           # rimuove ultima colonna (target)
    line = $0;
    sub(/[ \t]+$/, "", line);       # rimuove eventuale spazio finale
    print line, 1                   # aggiunge bias
}' "$INPUT_DATASET" > "$INPUT_FILE"

LAYER_INDEX=1
CURR_INPUT="$INPUT_FILE"
for LAYER_FILE in "$LAYERS_DIR"/layer*.txt; do
    OUTFILE="$TMPDIR/layer${LAYER_INDEX}.out"

    ACTIVATION_FUNCTION=$(grep '^ACTIVATION=' "$LAYER_FILE" | cut -d= -f2)
    NROWS=$(grep '^ROWS=' "$LAYER_FILE" | cut -d= -f2)
    NCOLS=$(grep '^COLS=' "$LAYER_FILE" | cut -d= -f2)

    awk -v weights="$LAYER_FILE" \
        -v activation_function="$ACTIVATION_FUNCTION" \
        -v nrows="$NROWS" \
        -v ncols="$NCOLS" \
        -v output="$OUTFILE" \
        -f lib/functions/math.awk \
        -f lib/functions/activation.awk \
        -f lib/functions/nnet-forward.awk \
        "$CURR_INPUT"

    # Prepara input per il prossimo layer, aggiungendo bias=1
    CURR_INPUT_WITH_BIAS="$TMPDIR/layer${LAYER_INDEX}_with_bias.tmp"
    awk '{ print $0, 1 }' "$OUTFILE" > "$CURR_INPUT_WITH_BIAS"
    CURR_INPUT="$CURR_INPUT_WITH_BIAS"

    ((LAYER_INDEX++))
done

# Output finale
cat "$OUTFILE"

