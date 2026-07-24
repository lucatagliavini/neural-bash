#!/bin/bash
#
# Dispatcher: test softmax + CCE multi-classe su iris_toy.
#

source "$(dirname "$0")/lib/framework.sh"

DATASET="$ROOT_DIR/dataset/iris_toy.txt"

suite_begin "Multiclass — nnet-init accetta softmax come activation"

_test_init_softmax() {
    local model
    model=$(make_tmp_model "mc_init" 4,8,3 --activation softmax --hidden-act sigmoid --method xavier --seed 7)
    assert_file_exists "layer1.txt creato con hidden=sigmoid" "$model/layer1.txt"
    assert_file_contains "layer1 ha ACTIVATION=sigmoid" "$model/layer1.txt" "^ACTIVATION=sigmoid$"
    assert_file_contains "layer2 ha ACTIVATION=softmax" "$model/layer2.txt" "^ACTIVATION=softmax$"
}

_test_init_softmax

suite_begin "Multiclass — convergenza softmax+CCE su iris_toy (3 classi)"

_test_softmax_cce_converges() {
    local model out
    model=$(make_tmp_model "mc_conv" 4,8,3 --activation softmax --hidden-act sigmoid --method xavier --seed 7)
    out=$("$NNET_RUN" train "$DATASET" "$model" \
        --epochs 3000 --optimizer adam --loss cce --lr 0.01 2>&1)
    assert_accuracy_above "softmax+CCE accuracy > 90%" 90 "$out"
}

_test_softmax_cce_converges

suite_begin "Multiclass — CCE loss decresce durante il training"

_test_cce_loss_decreases() {
    local model out loss_e1 loss_final
    model=$(make_tmp_model "mc_loss" 4,8,3 --activation softmax --hidden-act sigmoid --method xavier --seed 7)
    out=$("$NNET_RUN" train "$DATASET" "$model" \
        --epochs 1000 --optimizer adam --loss cce --lr 0.01 2>&1)
    loss_e1=$(echo    "$out" | grep "EPOCH 1\]"    | grep -oE "LOSS\(cce\) = [0-9.]+" | grep -oE "[0-9.]+$")
    loss_final=$(echo "$out" | grep "EPOCH 1000\]" | grep -oE "LOSS\(cce\) = [0-9.]+" | grep -oE "[0-9.]+$")
    if [[ -n "$loss_e1" && -n "$loss_final" ]]; then
        if awk "BEGIN { exit ($loss_final < $loss_e1) ? 0 : 1 }"; then
            _pass "CCE decresce: epoch1=$loss_e1 > epoch1000=$loss_final"
        else
            _fail "CCE deve decrescere" "epoch1=$loss_e1, epoch1000=$loss_final"
        fi
    else
        _fail "CCE decresce" "loss non trovata nell'output"
    fi
}

_test_cce_loss_decreases

suite_begin "Multiclass — predict mostra predizioni multi-classe"

_test_multiclass_predict() {
    local model out
    model=$(make_tmp_model "mc_pred" 4,8,3 --activation softmax --hidden-act sigmoid --method xavier --seed 7)
    "$NNET_RUN" train "$DATASET" "$model" --epochs 2000 --optimizer adam --loss cce --lr 0.01 > /dev/null 2>&1
    out=$("$NNET_RUN" predict "$DATASET" "$model" 2>&1)
    assert_match "predict produce PREDICTIONS"  "PREDICTIONS"        "$out"
    assert_match "predict produce Accuracy"     "Accuracy"           "$out"
    assert_match "output mostra 3 valori/riga"  "[0-9.]+ [0-9.]+ [0-9.]+" "$out"
}

_test_multiclass_predict

suite_begin "Multiclass — softmax output è distribuzione di probabilità (somma~1)"

_test_softmax_sum_to_one() {
    local model out sum_line
    model=$(make_tmp_model "mc_sum" 4,8,3 --activation softmax --hidden-act sigmoid --method xavier --seed 7)
    "$NNET_RUN" train "$DATASET" "$model" --epochs 2000 --optimizer adam --loss cce --lr 0.01 > /dev/null 2>&1
    # Prendi la prima riga di predizioni e somma i 3 valori
    local first_pred
    first_pred=$("$NNET_RUN" predict "$DATASET" "$model" 2>&1 | \
        grep -E "^[0-9]+ +\|" | head -1 | awk '{print $3, $4, $5}')
    local total
    total=$(echo "$first_pred" | awk '{s=$1+$2+$3; printf "%.4f", s}')
    if awk "BEGIN { exit ($total > 0.99 && $total < 1.01) ? 0 : 1 }"; then
        _pass "softmax somma a 1 (sum=$total)"
    else
        _fail "softmax somma a 1" "somma=$total (atteso ~1.0)"
    fi
}

_test_softmax_sum_to_one

suite_begin "Multiclass — --hidden-act default uguale a --activation"

_test_hidden_act_default() {
    local model
    model=$(make_tmp_model "mc_default" 4,4,3 --activation sigmoid --method xavier --seed 1)
    # Senza --hidden-act, tutti i layer devono avere sigmoid
    assert_file_contains "layer1 default sigmoid" "$model/layer1.txt" "^ACTIVATION=sigmoid$"
    assert_file_contains "layer2 default sigmoid" "$model/layer2.txt" "^ACTIVATION=sigmoid$"
}

_test_hidden_act_default

cleanup_tmp

suite_summary
