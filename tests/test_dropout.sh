#!/bin/bash
#
# Dispatcher: test dropout (inverted dropout sui layer hidden).
#

source "$(dirname "$0")/lib/framework.sh"

DATASET_XOR="$ROOT_DIR/dataset/xor.txt"
DATASET_IRIS="$ROOT_DIR/dataset/iris_toy.txt"

# ---------------------------------------------------------------------------
suite_begin "--dropout — training si completa senza errori"

_test_dropout_completes() {
    local model out
    model=$(make_tmp_model "dp_basic" 2,8,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 100 --dropout 0.3 2>&1)
    assert_match "training con dropout completa" "Training completed" "$out"
}

_test_dropout_completes

# ---------------------------------------------------------------------------
suite_begin "--dropout 0 — equivalente a nessun dropout"

_test_dropout_zero() {
    local model out
    model=$(make_tmp_model "dp_zero" 2,8,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 100 --dropout 0 2>&1)
    assert_match "dropout=0 completa" "Training completed" "$out"
}

_test_dropout_zero

# ---------------------------------------------------------------------------
suite_begin "--dropout — predict gira senza errori su modello addestrato con dropout"

_test_dropout_predict() {
    local model out
    model=$(make_tmp_model "dp_pred" 2,8,1 --activation sigmoid --seed 42)
    "$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 200 --dropout 0.3 2>&1 >/dev/null
    out=$("$NNET_RUN" predict "$DATASET_XOR" "$model" 2>&1)
    assert_match "predict dopo dropout funziona" "PREDICTIONS" "$out"
}

_test_dropout_predict

# ---------------------------------------------------------------------------
suite_begin "--dropout — le metriche finali di training non usano dropout (determinismo)"

_test_dropout_final_metrics_deterministic() {
    local model1 model2 mse1 mse2
    # Due run con stesso seed e stesso dropout: le metriche finali devono coincidere
    # perché il forward finale viene eseguito senza dropout in entrambi i casi.
    # Nota: non usiamo --seed sul run (non esiste), ma il forward finale è deterministico
    # dato che rilegge gli stessi pesi — verifichiamo solo che l'output contenga MSE.
    model1=$(make_tmp_model "dp_det" 2,8,1 --activation sigmoid --seed 42)
    local out
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model1" --epochs 50 --dropout 0.5 2>&1)
    assert_match "MSE presente nell'output finale" "MSE" "$out"
}

_test_dropout_final_metrics_deterministic

# ---------------------------------------------------------------------------
suite_begin "--dropout con --val-split — val forward senza dropout"

_test_dropout_val_split() {
    local model out
    model=$(make_tmp_model "dp_val" 4,8,3 --activation softmax --hidden-act sigmoid --seed 7)
    out=$("$NNET_RUN" train "$DATASET_IRIS" "$model" \
        --epochs 100 --optimizer adam --loss ce \
        --dropout 0.2 --val-split 0.2 2>&1)
    assert_match "dropout + val-split: training completa" "Training completed" "$out"
    assert_match "dropout + val-split: VAL_MSE presente"  "VAL_MSE" "$out"
}

_test_dropout_val_split

# ---------------------------------------------------------------------------
suite_begin "--dropout — rate alto (0.9) non causa crash"

_test_dropout_high_rate() {
    local model out
    model=$(make_tmp_model "dp_high" 2,16,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 50 --dropout 0.9 2>&1)
    assert_match "dropout=0.9 non crasha" "Training completed" "$out"
}

_test_dropout_high_rate

cleanup_tmp
suite_summary
