#!/bin/bash
#
# Dispatcher: test normalizzazione input (--normalize, z-score).
#

source "$(dirname "$0")/lib/framework.sh"

DATASET_XOR="$ROOT_DIR/dataset/xor.txt"
DATASET_IRIS="$ROOT_DIR/dataset/iris_toy.txt"
DATASET_SINE="$ROOT_DIR/dataset/sine_regression.txt"

# ---------------------------------------------------------------------------
suite_begin "--normalize — salva normalize.conf nel model_dir"

_test_conf_created() {
    local model
    model=$(make_tmp_model "norm_conf" 2,4,1 --activation sigmoid --seed 42)
    "$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 10 --normalize --no-save 2>&1 | grep -q "normalization enabled" || true
    if [[ -f "$model/normalize.conf" ]]; then
        _pass "normalize.conf creato"
    else
        _fail "normalize.conf non trovato" "file assente in $model"
    fi
}

_test_conf_created

# ---------------------------------------------------------------------------
suite_begin "--normalize — normalize.conf contiene mean e std per ogni feature"

_test_conf_content() {
    local model
    model=$(make_tmp_model "norm_content" 2,4,1 --activation sigmoid --seed 42)
    "$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 10 --normalize --no-save 2>&1 >/dev/null
    local conf="$model/normalize.conf"
    assert_match "num_inputs presente" "num_inputs=" "$(cat "$conf")"
    assert_match "mean_1 presente"     "mean_1="     "$(cat "$conf")"
    assert_match "std_1 presente"      "std_1="      "$(cat "$conf")"
    assert_match "mean_2 presente"     "mean_2="     "$(cat "$conf")"
    assert_match "std_2 presente"      "std_2="      "$(cat "$conf")"
}

_test_conf_content

# ---------------------------------------------------------------------------
suite_begin "--normalize — senza flag nessun normalize.conf"

_test_no_conf_without_flag() {
    local model
    model=$(make_tmp_model "norm_no" 2,4,1 --activation sigmoid --seed 42)
    "$NNET_RUN" train "$DATASET_XOR" "$model" --epochs 10 2>&1 >/dev/null
    if [[ -f "$model/normalize.conf" ]]; then
        _fail "normalize.conf NON deve esistere senza --normalize" "file trovato"
    else
        _pass "nessun normalize.conf senza --normalize"
    fi
}

_test_no_conf_without_flag

# ---------------------------------------------------------------------------
suite_begin "--normalize — predict rileva normalize.conf e lo applica"

_test_predict_applies_norm() {
    local model out
    model=$(make_tmp_model "norm_pred" 2,4,1 --activation sigmoid --seed 42)
    "$NNET_RUN" train "$DATASET_XOR" "$model" \
        --epochs 100 --normalize 2>&1 >/dev/null
    out=$("$NNET_RUN" predict "$DATASET_XOR" "$model" 2>&1)
    assert_match "predict applica z-score" "normalization applied" "$out"
}

_test_predict_applies_norm

# ---------------------------------------------------------------------------
suite_begin "--normalize — training si completa senza errori su iris_toy (4 feature)"

_test_normalize_iris() {
    local model out
    model=$(make_tmp_model "norm_iris" 4,8,3 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_IRIS" "$model" \
        --epochs 200 --optimizer adam --loss ce \
        --normalize 2>&1)
    assert_match "training completa" "Training completed" "$out"
    assert_match "normalize.conf iris" "normalization enabled" "$out"
}

_test_normalize_iris

# ---------------------------------------------------------------------------
suite_begin "--normalize — training si completa su sine_regression (task=regression)"

_test_normalize_sine() {
    local model out
    model=$(make_tmp_model "norm_sine" 1,8,1 --activation linear --seed 42)
    out=$("$NNET_RUN" train "$DATASET_SINE" "$model" \
        --epochs 100 --optimizer adam --task regression \
        --normalize 2>&1)
    assert_match "sine regression con normalize" "Training completed" "$out"
}

_test_normalize_sine

# ---------------------------------------------------------------------------
suite_begin "--normalize con --val-split — normalizzazione si applica anche al val set"

_test_normalize_val_split() {
    local model out
    model=$(make_tmp_model "norm_val" 4,8,3 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET_IRIS" "$model" \
        --epochs 100 --optimizer adam --loss ce \
        --normalize --val-split 0.2 2>&1)
    assert_match "normalize con val-split" "normalization enabled" "$out"
    assert_match "VAL_MSE presente" "VAL_MSE" "$out"
}

_test_normalize_val_split

cleanup_tmp
suite_summary
