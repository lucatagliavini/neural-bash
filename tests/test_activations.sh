#!/bin/bash
#
# Dispatcher: test di integrazione per le funzioni di attivazione.
# Verifica che ogni attivazione converga su XOR con seed fisso,
# e che il round-trip leaky_relu:alpha sopravviva a save/load.
#

source "$(dirname "$0")/lib/framework.sh"

DATASET="$ROOT_DIR/dataset/xor.txt"

suite_begin "Activations — convergenza su XOR (seed=42, 3000 epoche, adam)"

_test_activation_convergence() {
    local act="$1" mse_threshold="$2"; shift 2
    local model
    model=$(make_tmp_model "act_${act//:/_}" 2,4,1 --activation "$act" --seed 42 "$@")
    local out
    out=$("$NNET_RUN" eval "$DATASET" "$model" --epochs 3000 --optimizer adam 2>&1)
    assert_mse_below "convergenza $act" "$mse_threshold" "$out"
    assert_accuracy_above "accuracy $act" 75 "$out"
}

_test_activation_convergence "sigmoid"         0.20
_test_activation_convergence "tanh"            0.05
_test_activation_convergence "leaky_relu" 0.10 --alpha 0.1

suite_begin "Activations — round-trip leaky_relu:alpha"

# AWK serializza i numeri senza zeri trailing (0.10 → 0.1, 0.20 → 0.2).
# Il pattern di verifica usa il valore serializzato da AWK, non quello passato.
_test_leaky_roundtrip() {
    local alpha="$1" awk_alpha="$2"
    local model
    model=$(make_tmp_model "leaky_rt_${alpha//./_}" 2,4,1 --activation leaky_relu --alpha "$alpha" --seed 42)

    "$NNET_RUN" train "$DATASET" "$model" --epochs 100 --optimizer sgd > /dev/null 2>&1

    assert_file_contains "alpha=$alpha sopravvive in layer1 dopo train" \
        "$model/layer1.txt" "^ACTIVATION=leaky_relu:${awk_alpha}$"
    assert_file_contains "alpha=$alpha sopravvive in layer2 dopo train" \
        "$model/layer2.txt" "^ACTIVATION=leaky_relu:${awk_alpha}$"
    assert_file_contains "alpha=$alpha sopravvive in best/layer1" \
        "$model/best/layer1.txt" "^ACTIVATION=leaky_relu:${awk_alpha}$"
}

_test_leaky_roundtrip 0.01 0.01
_test_leaky_roundtrip 0.10 0.1
_test_leaky_roundtrip 0.20 0.2

cleanup_tmp

suite_summary
