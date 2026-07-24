#!/bin/bash
#
# Dispatcher: test best checkpoint, --use-best, --no-save.
#

source "$(dirname "$0")/lib/framework.sh"

DATASET="$ROOT_DIR/dataset/xor.txt"

suite_begin "Checkpoint — best/ creato dopo training"

_test_best_created() {
    local model
    model=$(make_tmp_model "ckpt_best" 2,4,1 --activation sigmoid --seed 42)
    "$NNET_RUN" train "$DATASET" "$model" --epochs 500 --optimizer adam > /dev/null 2>&1
    assert_file_exists "best/layer1.txt creato" "$model/best/layer1.txt"
    assert_file_exists "best/layer2.txt creato" "$model/best/layer2.txt"
}

_test_best_created

suite_begin "Checkpoint — best MSE <= MSE finale"

_test_best_le_final() {
    local model out best_mse final_mse
    model=$(make_tmp_model "ckpt_le" 2,4,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$DATASET" "$model" --epochs 1000 --optimizer sgd 2>&1)

    best_mse=$(echo  "$out" | grep -oE "best MSE = [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    final_mse=$(echo "$out" | grep "EPOCH 1000"   | grep -oE "MSE = [0-9.]+" | grep -oE "[0-9.]+$" | head -1)

    if [[ -n "$best_mse" && -n "$final_mse" ]]; then
        if awk "BEGIN { exit ($best_mse <= $final_mse) ? 0 : 1 }"; then
            _pass "best MSE ($best_mse) <= MSE finale ($final_mse)"
        else
            _fail "best MSE <= MSE finale" "best=$best_mse > final=$final_mse"
        fi
    else
        _fail "best MSE <= MSE finale" "valori non estratti dall'output"
    fi
}

_test_best_le_final

suite_begin "Checkpoint — --use-best carica da best/"

_test_use_best() {
    local model out_live out_best mse_live mse_best
    model=$(make_tmp_model "ckpt_usebest" 2,4,1 --activation sigmoid --seed 42)

    "$NNET_RUN" train "$DATASET" "$model" --epochs 2000 --optimizer adam > /dev/null 2>&1

    out_live=$( "$NNET_RUN" predict "$DATASET" "$model"            2>&1)
    out_best=$( "$NNET_RUN" predict "$DATASET" "$model" --use-best 2>&1)

    # Entrambi devono produrre output valido
    assert_match "--use-best produce MSE"      "Mean Squared Error" "$out_best"
    assert_match "--use-best produce Accuracy" "Accuracy"           "$out_best"

    # Il best checkpoint deve essere <= live (best è il minimo storico)
    mse_live=$(echo "$out_live" | grep -oE "Mean Squared Error \(MSE\) +: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    mse_best=$(echo "$out_best" | grep -oE "Mean Squared Error \(MSE\) +: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    if [[ -n "$mse_live" && -n "$mse_best" ]]; then
        if awk "BEGIN { exit ($mse_best <= $mse_live + 1e-9) ? 0 : 1 }"; then
            _pass "--use-best MSE ($mse_best) <= live MSE ($mse_live)"
        else
            _fail "--use-best MSE <= live MSE" "best=$mse_best > live=$mse_live"
        fi
    else
        _fail "--use-best MSE <= live MSE" "valori non estratti"
    fi
}

_test_use_best

suite_begin "Checkpoint — --no-save non modifica i pesi"

_test_no_save() {
    local model
    model=$(make_tmp_model "ckpt_nosave" 2,4,1 --activation sigmoid --seed 42)

    local before after
    before=$(md5sum "$model/layer1.txt" | cut -d' ' -f1)
    "$NNET_RUN" train "$DATASET" "$model" --epochs 200 --no-save > /dev/null 2>&1
    after=$(md5sum "$model/layer1.txt" | cut -d' ' -f1)

    assert_eq "--no-save: layer1.txt invariato" "$before" "$after"
}

_test_no_save

cleanup_tmp

suite_summary
