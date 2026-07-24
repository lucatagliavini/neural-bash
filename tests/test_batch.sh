#!/bin/bash
#
# Dispatcher: test mini-batch SGD (--batch-size N).
#

source "$(dirname "$0")/lib/framework.sh"

DATASET_XOR="$ROOT_DIR/dataset/xor.txt"
DATASET_IRIS="$ROOT_DIR/dataset/iris_toy.txt"

# ---------------------------------------------------------------------------
suite_begin "--batch-size — training si completa senza errori"

_test_batch_completes() {
    local model out
    model=$(make_tmp_model "mb_basic" 2,8,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 100 --batch-size 2 2>&1)
    assert_match "training con batch-size=2 completa" "Training completed" "$out"
}

_test_batch_completes

# ---------------------------------------------------------------------------
suite_begin "--batch-size — log indica mini-batch attivo"

_test_batch_log() {
    local model out
    model=$(make_tmp_model "mb_log" 2,8,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 10 --batch-size 2 2>&1)
    assert_match "log mini-batch presente" "mini-batch SGD enabled" "$out"
}

_test_batch_log

# ---------------------------------------------------------------------------
suite_begin "--batch-size 0 — equivalente a full-batch (nessun log mini-batch)"

_test_batch_zero_is_fullbatch() {
    local model out
    model=$(make_tmp_model "mb_zero" 2,8,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 10 --batch-size 0 2>&1)
    # Con batch_size=0 NON deve comparire il log mini-batch
    if echo "$out" | grep -q "mini-batch SGD enabled"; then
        _fail "batch-size=0 è full-batch" "log mini-batch non atteso"
    else
        _pass "batch-size=0 è full-batch (nessun log mini-batch)"
    fi
}

_test_batch_zero_is_fullbatch

# ---------------------------------------------------------------------------
suite_begin "--batch-size >= N — degrada silenziosamente a full-batch"

_test_batch_ge_n_is_fullbatch() {
    local model out
    model=$(make_tmp_model "mb_large" 2,8,1 --activation sigmoid --seed 42)
    # XOR ha 4 campioni; batch-size=10 >= 4 → full-batch
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 10 --batch-size 10 2>&1)
    if echo "$out" | grep -q "mini-batch SGD enabled"; then
        _fail "batch-size>=N è full-batch" "log mini-batch non atteso"
    else
        _pass "batch-size>=N degrada a full-batch"
    fi
}

_test_batch_ge_n_is_fullbatch

# ---------------------------------------------------------------------------
suite_begin "--batch-size — convergenza XOR con batch=2"

_test_batch_converges_xor() {
    local model out
    model=$(make_tmp_model "mb_conv2" 2,8,1 --activation sigmoid --seed 1)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 3000 --lr 0.5 --batch-size 2 2>&1)
    assert_mse_below "XOR converge con batch=2 (MSE<0.05)" 0.05 "$out"
}

_test_batch_converges_xor

# ---------------------------------------------------------------------------
suite_begin "--batch-size — convergenza XOR con batch=4 (full-batch)"

_test_batch_converges_xor_b4() {
    local model out
    model=$(make_tmp_model "mb_conv4" 2,8,1 --activation sigmoid --seed 1)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 3000 --lr 0.5 --batch-size 4 2>&1)
    assert_mse_below "XOR converge con batch=4 (MSE<0.05)" 0.05 "$out"
}

_test_batch_converges_xor_b4

# ---------------------------------------------------------------------------
suite_begin "--batch-size con Adam — training si completa"

_test_batch_adam() {
    local model out
    model=$(make_tmp_model "mb_adam" 2,8,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 500 --optimizer adam --batch-size 2 2>&1)
    assert_match "Adam + mini-batch completa" "Training completed" "$out"
}

_test_batch_adam

# ---------------------------------------------------------------------------
suite_begin "--batch-size con --dropout — training si completa"

_test_batch_dropout() {
    local model out
    model=$(make_tmp_model "mb_dp" 2,8,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 200 --batch-size 2 --dropout 0.2 2>&1)
    assert_match "mini-batch + dropout completa" "Training completed" "$out"
}

_test_batch_dropout

# ---------------------------------------------------------------------------
suite_begin "--batch-size con iris (multiclass, batch=8)"

_test_batch_iris() {
    local model out
    model=$(make_tmp_model "mb_iris" 4,12,3 --activation softmax --hidden-act sigmoid --seed 5)
    out=$("$NNET_RUN" train "$DATASET_IRIS" "$model" \
        --epochs 500 --optimizer adam --loss ce --batch-size 8 2>&1)
    assert_match "mini-batch iris completa" "Training completed" "$out"
    assert_match "mini-batch iris log presente" "mini-batch SGD enabled" "$out"
}

_test_batch_iris

cleanup_tmp
suite_summary
