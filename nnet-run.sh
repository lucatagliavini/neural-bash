#!/bin/bash
#
# Script completo per training e testing di una neural network in AWK
# 
# Usage:
#   ./nnet-run.sh <command> <dataset_file> <model_dir> [options]
#
# Commands:
#   train      Train a neural network
#   predict    Make predictions using a trained model
#   eval       Train and then evaluate the model
#
# Example:
#   ./nnet-run.sh train dataset/xor.txt models/xor --epochs 1000
#   ./nnet-run.sh predict dataset/xor.txt models/xor
#   ./nnet-run.sh eval dataset/xor.txt models/xor --epochs 2000
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
GRADIENT_CLIP="0.0"
LOSS_FUNCTION="mse"
MAX_EPOCHS=1000
SAVE_MODEL=1

# ============================================================================
# FUNZIONI DI UTILITÀ
# ============================================================================

function print_usage() {
    cat << EOF
Usage: $0 <command> <dataset_file> <model_dir> [options]

Commands:
  train      Train a neural network
  predict    Make predictions using a trained model  
  eval       Train and then evaluate the model

Positional Arguments:
  dataset_file          Path to the dataset
  model_dir            Directory containing the model layers

Training Options:
  --inputs N           Number of input features (default: 2)
  --layers N           Number of layers in the network (default: 2)
  --optimizer OPT      Optimizer, sets the following parameter accordingly
                       Values: sgd (default), sgd-momentum, sgd-momentum-decay
  --lr RATE            Learning rate (default: 0.3)
  --lr-decay RATE      Learning rate decay (default: 0.0 means no decay)
  --momentum M         Momentum coefficient (default: 0.0)
  --gradient-clip G    Value of max/min gradient clipping, must be positive (default: 0.0) [with 0.0 means disabled]
  --epochs N           Maximum number of training epochs (default: 1000)
  --no-save            Don't save the model after training
  --loss               Function for LOSS, [mse = default], if sigmoid activation [ce = cross-entropy] is possibile
  --debug FLAG         Enable debug output (forward|backward|update|network|metrics|all)

Prediction Options:
  --inputs N           Number of input features (default: 2)
  --layers N           Number of layers in the network (default: 2)

Global Options:
  -h, --help           Show this help message

Examples:
  # Train a model
  $0 train dataset/xor.txt models/xor --epochs 2000 --lr 0.5

  # Make predictions
  $0 predict dataset/xor.txt models/xor

  # Train and evaluate in one command
  $0 eval dataset/and.txt models/and --epochs 1000

  # Train with debug output
  $0 train dataset/or.txt models/or --epochs 500 --debug backward

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
    local mode="$1"
    local required_files=(
	"utils-math.awk"
        "utils-activation.awk"
        "utils-shared.awk"
        "utils-network.awk"
        "utils-forward.awk"
        "utils-loss.awk"
    )
    
    if [[ "$mode" == "train" || "$mode" == "eval" ]]; then
        required_files+=("utils-backward.awk" "utils-update.awk" "nnet-train.awk")
    fi
    
    if [[ "$mode" == "predict" || "$mode" == "eval" ]]; then
        required_files+=("nnet-predict.awk")
    fi
    
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
# FUNZIONI PRINCIPALI
# ============================================================================

function do_train() {
    echo "[INFO] Starting training..."
    echo "[INFO] Dataset: $DATASET_FILE"
    echo "[INFO] Model: $MODEL_DIR"
    echo "[INFO] Parameters: inputs=$NUM_INPUTS, layers=$NUM_LAYERS, lr=$LEARNING_RATE, lr-decay=$LR_DECAY,
                             loss=$LOSS_FUNCTION, momentum=$MOMENTUM, epochs=$MAX_EPOCHS"
    echo ""

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
	-v gradient_clip="$GRADIENT_CLIP" \
        -v max_epochs="$MAX_EPOCHS" \
        -v save_model="$SAVE_MODEL" \
        -v print_result=1 \
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
}

function do_predict() {
    echo "[INFO] Starting prediction..."
    echo "[INFO] Dataset: $DATASET_FILE"
    echo "[INFO] Model: $MODEL_DIR"
    echo ""

    awk \
        -v dataset_file="$DATASET_FILE" \
        -v num_inputs="$NUM_INPUTS" \
        -v model_dir="$MODEL_DIR" \
        -v num_layers="$NUM_LAYERS" \
 	-f "$LIB_DIR/utils-math.awk" \
        -f "$LIB_DIR/utils-activation.awk" \
        -f "$LIB_DIR/utils-loss.awk" \
        -f "$LIB_DIR/utils-shared.awk" \
        -f "$LIB_DIR/utils-network.awk" \
        -f "$LIB_DIR/utils-forward.awk" \
        -f "$LIB_DIR/nnet-predict.awk" \
        /dev/null
}

function do_eval() {
    echo "[INFO] Running training and evaluation..."
    echo ""
    do_train
    echo ""
    echo "[INFO] Now evaluating the trained model..."
    echo ""
    do_predict
}

# ============================================================================
# PARSING DEGLI ARGOMENTI
# ============================================================================

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift

if [[ "$COMMAND" != "train" && "$COMMAND" != "predict" && "$COMMAND" != "eval" ]]; then
    echo "[ERROR] Unknown command: $COMMAND" >&2
    echo "Available commands: train, predict, eval" >&2
    exit 1
fi

if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi

DATASET_FILE="$1"
MODEL_DIR="$2"
shift 2

DEBUG_FLAGS=""

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
	--gradient-clip)
	    GRADIENT_CLIP="$2"
	    shift 2
	    ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --no-save)
            SAVE_MODEL=0
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
        LEARNING_RATE="${LEARNING_RATE:=0.5}"
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
check_awk_files "$COMMAND"

# ============================================================================
# ESECUZIONE DEL COMANDO
# ============================================================================

case "$COMMAND" in
    train)
        do_train
        ;;
    predict)
        do_predict
        ;;
    eval)
        do_eval
        ;;
esac
