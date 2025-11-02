#!/bin/bash
#
# Script bash wrapper per nnet-init.awk
# Fornisce validazione e interfaccia user-friendly
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cerca nnet-init.awk
AWK_SCRIPT=""
if [ -f "$SCRIPT_DIR/nnet-init.awk" ]; then
    AWK_SCRIPT="$SCRIPT_DIR/nnet-init.awk"
elif [ -f "$SCRIPT_DIR/lib/framework/nnet-init.awk" ]; then
    AWK_SCRIPT="$SCRIPT_DIR/lib/framework/nnet-init.awk"
elif [ -f "./nnet-init.awk" ]; then
    AWK_SCRIPT="./nnet-init.awk"
else
    echo "[ERROR] Cannot find nnet-init.awk" >&2
    echo "Please ensure nnet-init.awk is in the same directory as this script" >&2
    exit 1
fi

# Parametri di default
ACTIVATION_FUNCTION="sigmoid"
INIT_METHOD="xavier"
SEED=""

function print_usage() {
    cat << EOF
Usage: $0 <model_dir> <architecture> [options]

Positional Arguments:
  model_dir            Directory where to save the model layers
  architecture         Network architecture (comma-separated layer sizes)
                       Format: input_size,hidden1_size,...,output_size
                       Example: 2,3,1 = 2 inputs, 3 hidden, 1 output

Options:
  --activation FUNC    Activation function (default: sigmoid)
                       Available: sigmoid, tanh, relu, leaky_relu
  --method METHOD      Weight initialization method (default: xavier)
                       Available: xavier, he, random
  --seed N             Random seed for reproducibility (optional)
  -h, --help           Show this help message

Examples:
  # XOR problem (2 inputs, 3 hidden, 1 output)
  $0 models/xor 2,3,1

  # Custom network with ReLU and He initialization
  $0 models/custom 4,8,8,2 --activation relu --method he

  # With reproducible seed
  $0 models/test 2,4,1 --seed 42

  # Multi-layer deep network
  $0 models/deep 10,20,20,10,5,2 --activation relu --method he

Architecture Examples:
  2,3,1       → XOR/simple logic gates (2 inputs, 3 hidden, 1 output)
  2,4,4,1     → 2-hidden layer network
  4,8,8,2     → 2-class classifier with 4 features
  10,20,10,5  → Deep network for complex problems

Initialization Methods:
  xavier      → Best for sigmoid/tanh (default)
  he          → Best for relu/leaky_relu
  random      → Simple uniform [-0.5, 0.5] (for testing)

EOF
}

function validate_architecture() {
    local arch="$1"
    
    # Verifica formato (numeri separati da virgole)
    if ! [[ "$arch" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
        echo "[ERROR] Invalid architecture format: $arch" >&2
        echo "Expected format: N1,N2,N3,... (e.g., 2,3,1)" >&2
        return 1
    fi
    
    # Converti in array
    IFS=',' read -ra LAYERS <<< "$arch"
    
    # Verifica almeno 2 layer (input + output)
    if [ ${#LAYERS[@]} -lt 2 ]; then
        echo "[ERROR] Architecture must have at least 2 layers (input and output)" >&2
        return 1
    fi
    
    # Verifica che ogni layer abbia almeno 1 neurone
    for size in "${LAYERS[@]}"; do
        if [ "$size" -lt 1 ]; then
            echo "[ERROR] Each layer must have at least 1 neuron" >&2
            return 1
        fi
    done
    
    return 0
}

function check_model_directory() {
    local model_dir="$1"
    
    if [ -d "$model_dir" ]; then
        echo "[WARNING] Directory $model_dir already exists." >&2
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "[INFO] Initialization cancelled." >&2
            exit 0
        fi
        echo "[INFO] Backing up existing model to ${model_dir}.backup" >&2
        rm -rf "${model_dir}.backup"
        cp -r "$model_dir" "${model_dir}.backup" 2>/dev/null || true
    fi
}

# ============================================================================
# PARSING DEGLI ARGOMENTI
# ============================================================================

if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi

MODEL_DIR="$1"
ARCHITECTURE="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --activation)
            ACTIVATION_FUNCTION="$2"
            shift 2
            ;;
        --method)
            INIT_METHOD="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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

echo "[INFO] Validating parameters..." >&2

# Valida architettura
if ! validate_architecture "$ARCHITECTURE"; then
    exit 1
fi

# Valida activation function
case "$ACTIVATION_FUNCTION" in
    sigmoid|tanh|relu|leaky_relu)
        ;;
    *)
        echo "[ERROR] Invalid activation function: $ACTIVATION_FUNCTION" >&2
        echo "Available: sigmoid, tanh, relu, leaky_relu" >&2
        exit 1
        ;;
esac

# Valida init method
case "$INIT_METHOD" in
    xavier|he|random)
        ;;
    *)
        echo "[ERROR] Invalid initialization method: $INIT_METHOD" >&2
        echo "Available: xavier, he, random" >&2
        exit 1
        ;;
esac

# Check directory esistente
check_model_directory "$MODEL_DIR"

# ============================================================================
# CHIAMATA A nnet-init.awk
# ============================================================================

echo "" >&2
echo "[INFO] Using AWK script: $AWK_SCRIPT" >&2
echo "" >&2

# Costruisci comando AWK
AWK_CMD="awk -f \"$AWK_SCRIPT\" \
    -v model_dir=\"$MODEL_DIR\" \
    -v architecture=\"$ARCHITECTURE\" \
    -v activation=\"$ACTIVATION_FUNCTION\" \
    -v init_method=\"$INIT_METHOD\""

# Aggiungi seed se fornito
if [ -n "$SEED" ]; then
    AWK_CMD="$AWK_CMD -v seed=$SEED"
fi

# Esegui
AWK_CMD="$AWK_CMD /dev/null"

eval $AWK_CMD

# ============================================================================
# POST-PROCESSING
# ============================================================================

# Verifica che i file siano stati creati
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "" >&2
    echo "[ERROR] Model initialization failed!" >&2
    echo "No files were created in $MODEL_DIR" >&2
    exit 1
fi

# Count layers creati
NUM_LAYERS=$(ls "$MODEL_DIR"/layer*.txt 2>/dev/null | wc -l)

if [ "$NUM_LAYERS" -eq 0 ]; then
    echo "" >&2
    echo "[ERROR] No layer files were created!" >&2
    exit 1
fi

# Estrai info per next steps
IFS=',' read -ra LAYER_SIZES <<< "$ARCHITECTURE"
NUM_INPUTS=${LAYER_SIZES[0]}

# ============================================================================
# SUMMARY AGGIUNTIVO
# ============================================================================

echo "" >&2
echo "==========================================" >&2
echo "SUCCESS! Model initialized." >&2
echo "==========================================" >&2
echo "" >&2
echo "Next steps:" >&2
echo "" >&2
echo "1. Train the model:" >&2
echo "   ./nnet-run.sh train dataset/your_data.txt $MODEL_DIR \\" >&2
echo "       --inputs $NUM_INPUTS \\" >&2
echo "       --layers $NUM_LAYERS \\" >&2
echo "       --epochs 1000 \\" >&2
echo "       --lr 0.3" >&2
echo "" >&2
echo "2. Or use eval for training + testing:" >&2
echo "   ./nnet-run.sh eval dataset/your_data.txt $MODEL_DIR \\" >&2
echo "       --inputs $NUM_INPUTS --layers $NUM_LAYERS" >&2
echo "" >&2
echo "3. Verify weights are different:" >&2
echo "   head -n 5 $MODEL_DIR/layer*.txt" >&2
echo "==========================================" >&2
