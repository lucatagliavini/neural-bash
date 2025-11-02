#!/bin/bash
#
# Script wrapper per la predizione con una neural network in AWK
# 
# Usage:
#   ./nnet-predict.sh <dataset_file> <model_dir> [options]
#
# Example:
#   ./nnet-predict.sh dataset/xor.txt models/xor
#

set -e  # Exit on error

# ============================================================================
# CONFIGURAZIONE DEFAULT
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/lib/framework"

# Parametri di default
NUM_INPUTS=2
NUM_LAYERS=2

# ============================================================================
# FUNZIONI DI UTILITÀ
# ============================================================================

function print_usage() {
    cat << EOF
Usage: $0 <dataset_file> <model_dir> [options]

Positional Arguments:
  dataset_file          Path to the dataset for prediction
  model_dir            Directory containing the trained model layers

Options:
  --inputs N           Number of input features (default: 2)
  --layers N           Number of layers in the network (default: 2)
  -h, --help           Show this help message

Examples:
  # Basic prediction
  $0 dataset/xor.txt models/xor

  # Custom network architecture
  $0 dataset/test.txt models/custom --inputs 3 --layers 3

EOF
}

function validate_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] File not found: $file" >&2
        exit 1
    fi
}

function validate_directory() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "[ERROR] Directory not found: $dir" >&2
        exit 1
    fi
}

function check_awk_files() {
    local required_files=(
        "utils-activation.awk"
        "utils-shared.awk"
        "utils-network.awk"
        "utils-forward.awk"
        "utils-metrics.awk"
        "nnet-predict.awk"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$LIB_DIR/$file" ]]; then
            echo "[ERROR] Required AWK file not found: $LIB_DIR/$file" >&2
            exit 1
        fi
    done
}

# ============================================================================
# PARSING DEGLI ARGOMENTI
# ============================================================================

if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi

DATASET_FILE="$1"
MODEL_DIR="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inputs)
            NUM_INPUTS="$2"
            shift 2
            ;;
        --layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# VALIDAZIONE
# ============================================================================

echo "[INFO] Validating configuration..."
validate_file "$DATASET_FILE"
validate_directory "$MODEL_DIR"
check_awk_files

# ============================================================================
# PREDIZIONE
# ============================================================================

# Esegui la predizione
awk \
    -v dataset_file="$DATASET_FILE" \
    -v num_inputs="$NUM_INPUTS" \
    -v model_dir="$MODEL_DIR" \
    -v num_layers="$NUM_LAYERS" \
    -f "$LIB_DIR/utils-activation.awk" \
    -f "$LIB_DIR/utils-shared.awk" \
    -f "$LIB_DIR/utils-network.awk" \
    -f "$LIB_DIR/utils-forward.awk" \
    -f "$LIB_DIR/utils-metrics.awk" \
    -f "$LIB_DIR/nnet-predict.awk" \
    /dev/null
