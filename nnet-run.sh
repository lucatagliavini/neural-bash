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
NUM_INPUTS=""
NUM_LAYERS=""
OPTIMIZER="sgd"
LEARNING_RATE=""
LR_DECAY=""
MOMENTUM=""
LOSS_FUNCTION="mse"
MAX_EPOCHS=1000
SAVE_MODEL=1
USE_BEST_CHECKPOINT=0
GRAD_CLIP=""
WEIGHT_DECAY=""
VAL_SPLIT="0"
PATIENCE=""
TASK="classification"
NORMALIZE=0
DROPOUT=0
BATCH_SIZE=0
SEED=""

# ============================================================================
# FUNZIONI DI UTILITÀ
# ============================================================================

# Legge il valore di una chiave da un file key=value.
# Uso: read_conf_key <file> <chiave>
function read_conf_key() {
    grep "^$2=" "$1" 2>/dev/null | cut -d= -f2
}

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
  --inputs N           Number of input features (auto-detected from model, default: 2)
  --layers N           Number of layers in the network (auto-detected from model, default: 2)
  --optimizer OPT      Optimizer, sets the following parameter accordingly
                       Values: sgd (default), sgd-momentum, sgd-momentum-decay
  --lr RATE            Learning rate (default: 0.3)
  --lr-decay RATE      Learning rate decay (default: 0.0 means no decay)
  --momentum M         Momentum coefficient (default: 0.0)
  --epochs N           Maximum number of training epochs (default: 1000)
  --no-save            Don't save the model after training
  --use-best           Use the best checkpoint (model_dir/best/) instead of the live weights
  --loss               Function for LOSS, [mse = default], if sigmoid activation [ce = cross-entropy] is possibile
  --task TASK          Task type: classification (default) or regression
  --val-split N        Fraction of data for validation set (e.g. 0.2, default: 0)
  --patience N         Early stopping: stop if val_mse doesn't improve for N epochs (requires --val-split)
  --clip N             Gradient clipping threshold (default: disabled)
  --wd N               L2 weight decay coefficient (default: 0.0)
  --normalize          Normalize input features with z-score (mean=0, std=1) computed on training data
  --dropout N          Dropout rate for hidden layers during training (e.g. 0.2, default: 0)
  --batch-size N       Mini-batch size (default: 0 = full-batch); set to 2-N for mini-batch SGD
  --seed N             Random seed for reproducibility (shuffle, dropout); default: time-based
  --debug FLAG         Enable debug output (forward|backward|update|network|metrics|all)

Prediction Options:
  --inputs N           Number of input features (auto-detected from model, default: 2)
  --layers N           Number of layers in the network (auto-detected from model, default: 2)

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
    echo "[INFO] Parameters: inputs=$NUM_INPUTS, layers=$NUM_LAYERS, lr=$LEARNING_RATE, lr-decay=$LR_DECAY,"
    echo "                   loss=$LOSS_FUNCTION, momentum=$MOMENTUM, epochs=$MAX_EPOCHS, clip=${GRAD_CLIP:-off}, wd=${WEIGHT_DECAY:-0}, batch=${BATCH_SIZE:-0}"
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
        -v max_epochs="$MAX_EPOCHS" \
        -v save_model="$SAVE_MODEL" \
        -v print_result=1 \
        -v task="${TASK:-classification}" \
        -v val_split="${VAL_SPLIT:-0}" \
        -v patience="${PATIENCE:-0}" \
        -v grad_clip="${GRAD_CLIP:-0}" \
        -v weight_decay="${WEIGHT_DECAY:-0}" \
        -v normalize="${NORMALIZE:-0}" \
        -v dropout="${DROPOUT:-0}" \
        -v batch_size="${BATCH_SIZE:-0}" \
        -v seed="${SEED:-0}" \
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

    # Aggiorna model.conf con i metadati dell'ultima sessione di training
    local meta_file="${MODEL_DIR}/.train_meta"
    if [[ -f "$meta_file" ]]; then
        local arch activation init_method
        # Leggi valori esistenti da model.conf (potrebbero non esserci)
        arch=$(       read_conf_key "$MODEL_DIR/model.conf" architecture)
        activation=$( read_conf_key "$MODEL_DIR/model.conf" activation)
        init_method=$(read_conf_key "$MODEL_DIR/model.conf" init_method)
        # Leggi i nuovi valori scritti da AWK
        source "$meta_file"
        cat > "$MODEL_DIR/model.conf" << CONFEOF
architecture=${arch:-unknown}
activation=${activation:-unknown}
init_method=${init_method:-unknown}
last_optimizer=${last_optimizer}
last_lr=${last_lr}
last_lr_decay=${last_lr_decay}
last_momentum=${last_momentum}
last_epochs=${last_epochs}
last_loss=${last_loss}
best_mse=${best_mse}
best_epoch=${best_epoch}
CONFEOF
        rm -f "$meta_file"
    fi
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
        -v task="${TASK:-classification}" \
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
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --val-split)
            VAL_SPLIT="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --clip)
            GRAD_CLIP="$2"
            shift 2
            ;;
        --wd)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --normalize)
            NORMALIZE=1
            shift
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-save)
            SAVE_MODEL=0
            shift
            ;;
        --use-best)
            USE_BEST_CHECKPOINT=1
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
# AUTO-DETECT num_inputs AND num_layers FROM MODEL FILES
# ============================================================================
if [[ -d "$MODEL_DIR" ]]; then
    _detected_layers=$(ls "$MODEL_DIR"/layer*.txt 2>/dev/null | wc -l | tr -d ' ')
    _layer1="$MODEL_DIR/layer1.txt"
    if [[ -f "$_layer1" ]]; then
        _detected_inputs=$(awk 'NR==2{print NF-1; exit}' "$_layer1")
    fi
    if [[ -z "$NUM_LAYERS" ]] && [[ "${_detected_layers:-0}" -gt 0 ]]; then
        NUM_LAYERS="$_detected_layers"
    fi
    if [[ -z "$NUM_INPUTS" ]] && [[ -n "${_detected_inputs:-}" ]]; then
        NUM_INPUTS="$_detected_inputs"
    fi
fi
: "${NUM_LAYERS:=2}"
: "${NUM_INPUTS:=2}"

# ============================================================================
# BEST CHECKPOINT OVERRIDE
# ============================================================================

if [[ "$USE_BEST_CHECKPOINT" == "1" ]]; then
    MODEL_DIR="${MODEL_DIR}/best"
    echo "[INFO] Using best checkpoint: $MODEL_DIR"
fi

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
