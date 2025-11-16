#!/bin/bash
#
# Script wrapper per il training di una neural network in AWK
# 
# Usage:
#   ./nnet-train.sh <dataset_file> <model_dir> [options]
#
# Example:
#   ./nnet-train.sh dataset/xor.txt models/xor --epochs 1000 --lr 0.3
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
OPTIMIZER="sgd"
LEARNING_RATE=""
LR_DECAY=""
MOMENTUM=""
LOSS_FUNCTION="mse"
MAX_EPOCHS=1000
PRINT_RESULT=0
DEBUG_FLAGS=""

# ============================================================================
# FUNZIONI DI UTILITÀ
# ============================================================================

function print_usage() {
    cat << EOF
Usage: $0 <dataset_file> <model_dir> [options]

Positional Arguments:
  dataset_file          Path to the training dataset
  model_dir            Directory containing the model layers

Options:
  --inputs N           Number of input features (default: 2)
  --layers N           Number of layers in the network (default: 2)
  --optimizer OPT      Optimizer, sets the following parameter accordingly
                       Values: sgd (default), sgd-momentum, sgd-momentum-decay
  --lr RATE            Learning rate (default: 0.3)
  --lr-decay RATE      Learning rate decay (default: 0.0 means no decay) 
  --loss LOSS          Loss function (default: mse) values: mse | ce
  --momentum M         Momentum coefficient (default: 0.0)
  --epochs N           Maximum number of training epochs (default: 1000)
  --print-result       Print predictions after training
  --debug FLAG         Enable debug output (forward|backward|update|network|metrics|all)
  -h, --help           Show this help message

Examples:
  # Basic training
  $0 dataset/xor.txt models/xor

  # Custom parameters with result printing
  $0 dataset/and.txt models/and --epochs 2000 --lr 0.5 --print-result

  # Training with debug output
  $0 dataset/or.txt models/or --epochs 500 --debug backward

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
	    "utils-math.awk"
        "utils-activation.awk"
        "utils-shared.awk"
        "utils-network.awk"
        "utils-forward.awk"
        "utils-backward.awk"
        "utils-update.awk"
        "utils-loss.awk"
        "nnet-train.awk"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$LIB_DIR/$file" ]]; then
            echo "[ERROR] Required AWK file not found: $LIB_DIR/$file" >&2
            exit 1
        fi
    done
}

function setup_debug_flags() {
    local debug_type="$1"
    
    case "$debug_type" in
        forward)
            DEBUG_FLAGS="-v debug_forward=1"
            ;;
        backward)
            DEBUG_FLAGS="-v debug_backward=1"
            ;;
        update)
            DEBUG_FLAGS="-v debug_update=1"
            ;;
        network)
            DEBUG_FLAGS="-v debug_network=1"
            ;;
        metrics)
            DEBUG_FLAGS="-v debug_metrics=1"
            ;;
        all)
            DEBUG_FLAGS="-v debug_forward=1 -v debug_backward=1 -v debug_update=1 -v debug_network=1 -v debug_metrics=1"
            ;;
        *)
            echo "[WARNING] Unknown debug flag: $debug_type. Available: forward|backward|update|network|metrics|all" >&2
            ;;
    esac
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
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lr-decay)
            LR_DECAY="$2"
            shift 2
            ;;
        --loss)
            LOSS_FUNCTION="$2"
            shift 2
            ;;
        --momentum)
            MOMENTUM="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --print-result)
            PRINT_RESULT=1
            shift
            ;;
        --debug)
            setup_debug_flags "$2"
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
# Applico l'OPTIMIZER:
# ============================================================================
case "${OPTIMIZER}" in
    sgd)
        : "${LEARNING_RATE:=0.3}"
        : "${MOMENTUM:=0.0}"
        : "${LR_DECAY:=0.0}"
        ;;
    sgd-momentum)
        : "${LEARNING_RATE:=0.5}"
        : "${MOMENTUM:=0.9}"
        : "${LR_DECAY:=0.0}"
        ;;
    sgd-momentum-decay)
        : "${LEARNING_RATE:=0.5}"
        : "${MOMENTUM:=0.9}"
        : "${LR_DECAY:=0.001}"
        ;;
    adam)
        : "${LEARNING_RATE:=0.001}"
        : "${MOMENTUM:=0.0}"     # ignorato
        : "${LR_DECAY:=0.0}"
        ;;
esac

# ============================================================================
# VALIDAZIONE
# ============================================================================

echo "[INFO] Validating configuration..."
validate_file "$DATASET_FILE"
validate_directory "$MODEL_DIR"
check_awk_files

# ============================================================================
# TRAINING
# ============================================================================

echo "[INFO] Starting training..."
echo "[INFO] Dataset: $DATASET_FILE"
echo "[INFO] Model: $MODEL_DIR"
echo "[INFO] Parameters: inputs=$NUM_INPUTS, layers=$NUM_LAYERS, lr=$LEARNING_RATE, lr-decay=$LR_DECAY,
                         loss=$LOSS_FUNCTION momentum=$MOMENTUM, epochs=$MAX_EPOCHS"
echo ""

# Esegui il training
awk \
    -v dataset_file="$DATASET_FILE" \
    -v num_inputs="$NUM_INPUTS" \
    -v model_dir="$MODEL_DIR" \
    -v num_layers="$NUM_LAYERS" \
    -v optimizer="$OPTIMIZER" \
    -v learning_rate="$LEARNING_RATE" \
    -v lr_decay="$LR_DECAY" \
    -v loss_function="$LOSS_FUNCTION" \
    -v momentum="$MOMENTUM" \
    -v max_epochs="$MAX_EPOCHS" \
    -v print_result="$PRINT_RESULT" \
    $DEBUG_FLAGS \
    -f "$LIB_DIR/utils-math.awk" \
    -f "$LIB_DIR/utils-activation.awk" \
    -f "$LIB_DIR/utils-loss.awk" \
    -f "$LIB_DIR/utils-shared.awk" \
    -f "$LIB_DIR/utils-network.awk" \
    -f "$LIB_DIR/utils-forward.awk" \
    -f "$LIB_DIR/utils-backward.awk" \
    -f "$LIB_DIR/utils-update.awk" \
    -f "$LIB_DIR/nnet-train.awk" \
    /dev/null

echo ""
echo "[INFO] Training completed!"
