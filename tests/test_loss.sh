#!/bin/bash
#
# Dispatcher: test funzioni di loss (MSE e CE).
#

source "$(dirname "$0")/lib/framework.sh"

DATASET="$ROOT_DIR/dataset/xor.txt"

suite_begin "Loss — MSE converge su XOR"

_test_loss_mse() {
    local model
    model=$(make_tmp_model "loss_mse" 2,4,1 --activation sigmoid --seed 42)
    local out
    out=$("$NNET_RUN" eval "$DATASET" "$model" --epochs 3000 --optimizer adam --loss mse 2>&1)
    # sigmoid+adam classifica correttamente ma MSE rimane ~0.14 per saturazione
    assert_mse_below      "mse: MSE < 0.20"      0.20 "$out"
    assert_accuracy_above "mse: accuracy = 100%"  99   "$out"
}

_test_loss_mse

suite_begin "Loss — CE converge su XOR (sigmoid)"

_test_loss_ce() {
    local model
    model=$(make_tmp_model "loss_ce" 2,4,1 --activation sigmoid --seed 42)
    local out
    out=$("$NNET_RUN" eval "$DATASET" "$model" --epochs 3000 --optimizer adam --loss ce 2>&1)
    assert_mse_below      "ce: MSE < 0.20"      0.20 "$out"
    assert_accuracy_above "ce: accuracy = 100%"  99   "$out"
}

_test_loss_ce

suite_begin "Loss — CE raggiunge accuracy = 100% prima di MSE (robustezza, 5 run)"

_test_ce_better_accuracy() {
    local wins=0 runs=5
    for i in $(seq 1 $runs); do
        local m_mse m_ce out_mse out_ce acc_mse acc_ce
        m_mse=$(make_tmp_model "loss_cmp_mse_$i" 2,4,1 --activation sigmoid)
        m_ce=$( make_tmp_model "loss_cmp_ce_$i"  2,4,1 --activation sigmoid)
        out_mse=$("$NNET_RUN" eval "$DATASET" "$m_mse" --epochs 1000 --optimizer adam --loss mse 2>&1)
        out_ce=$( "$NNET_RUN" eval "$DATASET" "$m_ce"  --epochs 1000 --optimizer adam --loss ce  2>&1)
        acc_mse=$(echo "$out_mse" | grep -oE "Accuracy +: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
        acc_ce=$( echo "$out_ce"  | grep -oE "Accuracy +: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
        if awk "BEGIN { exit ($acc_ce >= $acc_mse) ? 0 : 1 }"; then
            wins=$((wins + 1))
        fi
    done
    if [[ $wins -ge 3 ]]; then
        _pass "CE accuracy >= MSE-loss in $wins/$runs run"
    else
        _fail "CE accuracy >= MSE-loss" "solo $wins/$runs run favorevoli"
    fi
}

_test_ce_better_accuracy

suite_begin "Loss — CE con attivazione non-sigmoid fa fallback a MSE"

_test_ce_fallback() {
    local model
    model=$(make_tmp_model "loss_ce_tanh" 2,4,1 --activation tanh --seed 42)
    local out
    # Con tanh + CE il framework usa MSE internamente — non deve crashare
    out=$("$NNET_RUN" train "$DATASET" "$model" --epochs 500 --optimizer adam --loss ce 2>&1)
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        _pass "ce + tanh: nessun crash"
    else
        _fail "ce + tanh: nessun crash" "exit code=$exit_code"
    fi
}

_test_ce_fallback

cleanup_tmp

suite_summary
