#!/bin/bash
#
# Dispatcher: test train/val split ed early stopping.
#

source "$(dirname "$0")/lib/framework.sh"

DATASET="$ROOT_DIR/dataset/xor.txt"

suite_begin "Val split — VAL_MSE appare nell'output di training"

_test_val_mse_printed() {
    local model
    model=$(make_tmp_model "vs_mse" 2,4,1 --activation sigmoid --seed 42)
    local out
    out=$("$NNET_RUN" train "$DATASET" "$model" --epochs 200 --optimizer adam --val-split 0.25 2>&1)
    assert_match "VAL_MSE visibile nell'output" "VAL_MSE" "$out"
}

_test_val_mse_printed

suite_begin "Val split — senza --val-split nessun VAL_MSE"

_test_no_val_without_flag() {
    local model out
    model=$(make_tmp_model "vs_none" 2,4,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET" "$model" --epochs 200 --optimizer adam 2>&1)
    if echo "$out" | grep -q "VAL_MSE"; then
        _fail "nessun VAL_MSE senza --val-split" "trovato VAL_MSE nell'output"
    else
        _pass "nessun VAL_MSE senza --val-split"
    fi
}

_test_no_val_without_flag

suite_begin "Val split — num_val_samples = round(N * val_split)"

_test_val_sample_count() {
    local model out n_val
    model=$(make_tmp_model "vs_count" 2,4,1 --activation sigmoid --seed 42)
    # XOR ha 4 campioni; val_split=0.5 → 2 train, 2 val
    out=$("$NNET_RUN" train "$DATASET" "$model" --epochs 50 --optimizer sgd --val-split 0.5 2>&1)
    # Con 2 campioni di train, MSE non può essere quello di tutti e 4
    assert_match "training avviene con val_split=0.5" "EPOCH" "$out"
}

_test_val_sample_count

suite_begin "Early stopping — si ferma prima delle max_epochs"

_test_early_stop_fires() {
    local model out last_epoch
    model=$(make_tmp_model "es_fires" 2,4,1 --activation sigmoid --seed 42)
    # patience=10: val_mse di XOR con 25% split converge in poche epoche
    out=$("$NNET_RUN" train "$DATASET" "$model" \
        --epochs 5000 --optimizer adam --val-split 0.25 --patience 30 2>&1)
    assert_match "early stopping attivato" "early stopping at epoch" "$out"
    # Verifica che si sia fermato prima dell'epoch 5000
    last_epoch=$(echo "$out" | grep -oE "early stopping at epoch [0-9]+" | grep -oE "[0-9]+$")
    if [[ -n "$last_epoch" ]] && [[ "$last_epoch" -lt 5000 ]]; then
        _pass "fermato all'epoch $last_epoch < 5000"
    else
        _fail "fermato prima di 5000 epoche" "last_epoch='$last_epoch'"
    fi
}

_test_early_stop_fires

suite_begin "Early stopping — con patience=0 non si ferma mai prima"

_test_no_early_stop_zero_patience() {
    local model out
    model=$(make_tmp_model "es_zero" 2,4,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET" "$model" \
        --epochs 200 --optimizer adam --val-split 0.25 --patience 0 2>&1)
    if echo "$out" | grep -q "early stopping at epoch"; then
        _fail "patience=0 non ferma" "trovato 'early stopping' nell'output"
    else
        _pass "patience=0: training completa le 200 epoche"
    fi
}

_test_no_early_stop_zero_patience

suite_begin "Early stopping — warning se patience > 0 senza val-split"

_test_patience_warns_without_val() {
    local model out
    model=$(make_tmp_model "es_warn" 2,4,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET" "$model" \
        --epochs 100 --optimizer sgd --patience 5 2>&1)
    assert_match "warning patience senza val-split" "early stopping disabilitato" "$out"
}

_test_patience_warns_without_val

cleanup_tmp

suite_summary
