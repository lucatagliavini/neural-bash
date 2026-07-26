#!/bin/bash
#
# Hyperparameter search: lancia N training in parallelo su combinazioni
# di architetture, learning rate e optimizer, stampa un ranking per best_mse.
#
# Usage:
#   ./nnet-search.sh <dataset> <model_prefix> [options]
#
# Examples:
#   ./nnet-search.sh dataset/xor.txt /tmp/search \
#       --arch "2,4,1 2,8,1" --lr "0.01 0.1" --optimizer "sgd adam" --epochs 1000
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNET_INIT="$SCRIPT_DIR/nnet-init.sh"
NNET_RUN="$SCRIPT_DIR/nnet-run.sh"

# ============================================================================
# DEFAULT
# ============================================================================

ARCHS=""
LRS="0.01 0.1"
OPTIMIZERS="sgd adam"
EPOCHS=1000
MAX_JOBS=4
EXTRA_FLAGS=""

# ============================================================================
# USAGE
# ============================================================================

function print_usage() {
    cat << EOF
Usage: $0 <dataset> <model_prefix> [options]

Positional Arguments:
  dataset        Path to training dataset
  model_prefix   Directory prefix for temporary models (e.g. /tmp/search)

Options:
  --arch "A B"       Space-separated list of architectures (required)
  --lr "X Y"         Space-separated learning rates (default: "0.01 0.1")
  --optimizer "X Y"  Space-separated optimizers    (default: "sgd adam")
  --epochs N         Epochs per run                (default: 1000)
  --jobs N           Max parallel jobs             (default: 4)

Any additional flags (e.g. --loss ce --normalize) are forwarded to nnet-run.sh.

Example:
  $0 dataset/iris_toy.txt /tmp/iris_search \\
      --arch "4,8,3 4,16,3" --lr "0.001 0.01 0.1" \\
      --optimizer "sgd adam" --epochs 2000
EOF
}

# ============================================================================
# ARGOMENTI
# ============================================================================

if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi

DATASET="$1"
MODEL_PREFIX="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)      ARCHS="$2";      shift 2 ;;
        --lr)        LRS="$2";        shift 2 ;;
        --optimizer) OPTIMIZERS="$2"; shift 2 ;;
        --epochs)    EPOCHS="$2";     shift 2 ;;
        --jobs)      MAX_JOBS="$2";   shift 2 ;;
        -h|--help)   print_usage; exit 0 ;;
        *)           EXTRA_FLAGS="$EXTRA_FLAGS $1 $2"; shift 2 ;;
    esac
done

if [[ -z "$ARCHS" ]]; then
    echo "[ERROR] --arch è obbligatorio" >&2
    print_usage
    exit 1
fi

if [[ ! -f "$DATASET" ]]; then
    echo "[ERROR] Dataset non trovato: $DATASET" >&2
    exit 1
fi

# ============================================================================
# GENERAZIONE COMBINAZIONI
# ============================================================================

COMBOS=()
for arch in $ARCHS; do
    for lr in $LRS; do
        for opt in $OPTIMIZERS; do
            COMBOS+=("${arch}|${lr}|${opt}")
        done
    done
done

TOTAL=${#COMBOS[@]}
echo "[INFO] search: ${TOTAL} combinazioni, max ${MAX_JOBS} job paralleli"
echo "[INFO] search: dataset=${DATASET}, epochs=${EPOCHS}"
echo ""

# ============================================================================
# DIRECTORY RISULTATI
# ============================================================================

SEARCH_DIR="${MODEL_PREFIX}_search_$$"
mkdir -p "$SEARCH_DIR"

# File di riepilogo risultati (uno per combinazione, scritto dal subshell)
RESULTS_DIR="$SEARCH_DIR/results"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# FUNZIONE DI UN SINGOLO RUN
# ============================================================================

run_combo() {
    local idx="$1" arch="$2" lr="$3" opt="$4"
    local tag
    tag=$(printf "%03d" "$idx")
    local model_dir="$SEARCH_DIR/model_${tag}_${arch//,/-}_lr${lr}_${opt}"
    local result_file="$RESULTS_DIR/${tag}.txt"

    # Init + train (--no-save mantiene i pesi nel modello temporaneo)
    "$NNET_INIT" "$model_dir" "$arch" --force > /dev/null 2>&1

    local train_out
    if train_out=$("$NNET_RUN" train "$DATASET" "$model_dir" \
            --epochs "$EPOCHS" --lr "$lr" --optimizer "$opt" \
            --no-save $EXTRA_FLAGS 2>&1); then
        # Leggi best_mse da model.conf (scritto da nnet-run.sh anche con --no-save)
        local best_mse best_epoch
        best_mse=$(grep '^best_mse='    "$model_dir/model.conf" 2>/dev/null | cut -d= -f2)
        best_epoch=$(grep '^best_epoch=' "$model_dir/model.conf" 2>/dev/null | cut -d= -f2)
        best_mse="${best_mse:-N/A}"
        best_epoch="${best_epoch:-N/A}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$tag" "$arch" "$lr" "$opt" "$best_mse" "$best_epoch" > "$result_file"
    else
        printf "%s\t%s\t%s\t%s\tFAILED\tN/A\n" \
            "$tag" "$arch" "$lr" "$opt" > "$result_file"
    fi
}

# ============================================================================
# ESECUZIONE PARALLELA CON CAP
# ============================================================================

pids=()
idx=0

for combo in "${COMBOS[@]}"; do
    IFS='|' read -r arch lr opt <<< "$combo"
    idx=$((idx + 1))

    run_combo "$idx" "$arch" "$lr" "$opt" &
    pids+=($!)

    # Quando raggiungiamo il cap, aspettiamo il batch corrente
    if [[ ${#pids[@]} -ge $MAX_JOBS ]]; then
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
        pids=()
    fi
done

# Aspetta i job rimanenti
for pid in "${pids[@]}"; do
    wait "$pid" || true
done

# ============================================================================
# RANKING
# ============================================================================

echo ""
echo "══════════════════════════════════════════════════════════════"
printf "%-4s  %-12s  %-8s  %-20s  %-12s  %s\n" \
    "RANK" "ARCH" "LR" "OPTIMIZER" "BEST_MSE" "BEST_EPOCH"
echo "──────────────────────────────────────────────────────────────"

# Ordina per best_mse numericamente (FAILED va in fondo)
rank=0
while IFS=$'\t' read -r tag arch lr opt best_mse best_epoch; do
    rank=$((rank + 1))
    printf "%-4s  %-12s  %-8s  %-20s  %-12s  %s\n" \
        "$rank" "$arch" "$lr" "$opt" "$best_mse" "$best_epoch"
done < <(
    # Ordina numericamente per best_mse; FAILED (non numerico) va in fondo.
    # Insertion sort POSIX (no asorti gawk-specific).
    cat "$RESULTS_DIR"/*.txt | awk -F'\t' '
        $5 == "FAILED" { failed[++nf] = $0; next }
        { ok[++n] = $0; key[n] = $5+0 }
        END {
            for (i = 2; i <= n; i++) {
                tmp_line = ok[i]; tmp_key = key[i]
                j = i - 1
                while (j >= 1 && key[j] > tmp_key) {
                    ok[j+1] = ok[j]; key[j+1] = key[j]; j--
                }
                ok[j+1] = tmp_line; key[j+1] = tmp_key
            }
            for (i = 1; i <= n;  i++) print ok[i]
            for (i = 1; i <= nf; i++) print failed[i]
        }
    '
)

echo "══════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# CLEANUP
# ============================================================================

rm -rf "$SEARCH_DIR"
