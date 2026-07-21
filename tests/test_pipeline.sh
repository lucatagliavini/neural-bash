#!/bin/bash
#
# Dispatcher: test end-to-end della pipeline completa.
# Verifica init → train → predict → eval su XOR, AND, OR
# con topologie diverse e il comando eval combinato.
#

source "$(dirname "$0")/lib/framework.sh"

suite_begin "Pipeline — XOR (problema non linearmente separabile)"

_test_xor() {
    local model
    model=$(make_tmp_model "pipe_xor" 2,4,1 --activation sigmoid --method xavier --seed 42)
    local out
    out=$("$NNET_RUN" eval "$ROOT_DIR/dataset/xor.txt" "$model" \
        --epochs 3000 --optimizer adam --loss ce 2>&1)
    assert_accuracy_above "XOR accuracy > 99%" 99 "$out"
}

_test_xor

suite_begin "Pipeline — AND (problema linearmente separabile)"

_test_and() {
    local model
    model=$(make_tmp_model "pipe_and" 2,2,1 --activation sigmoid --seed 42)
    local out
    out=$("$NNET_RUN" eval "$ROOT_DIR/dataset/and.txt" "$model" \
        --epochs 1000 --optimizer sgd-momentum 2>&1)
    assert_accuracy_above "AND accuracy = 100%" 99 "$out"
}

_test_and

suite_begin "Pipeline — OR (problema linearmente separabile)"

_test_or() {
    local model
    model=$(make_tmp_model "pipe_or" 2,2,1 --activation sigmoid --seed 42)
    local out
    out=$("$NNET_RUN" eval "$ROOT_DIR/dataset/or.txt" "$model" \
        --epochs 1000 --optimizer sgd-momentum 2>&1)
    assert_accuracy_above "OR accuracy = 100%" 99 "$out"
}

_test_or

suite_begin "Pipeline — rete deep (3 layer nascosti)"

_test_deep() {
    local model
    model=$(make_tmp_model "pipe_deep" 2,8,4,1 --activation sigmoid --seed 42)
    local out
    out=$("$NNET_RUN" eval "$ROOT_DIR/dataset/xor.txt" "$model" \
        --epochs 3000 --optimizer adam 2>&1)
    assert_accuracy_above "deep XOR accuracy > 75%" 75 "$out"
}

_test_deep

suite_begin "Pipeline — predict separato da train"

_test_predict_separate() {
    local model out_train out_predict mse_train mse_predict
    model=$(make_tmp_model "pipe_sep" 2,4,1 --activation sigmoid --seed 42)

    "$NNET_RUN" train "$ROOT_DIR/dataset/xor.txt" "$model" \
        --epochs 2000 --optimizer adam > /dev/null 2>&1

    out_predict=$("$NNET_RUN" predict "$ROOT_DIR/dataset/xor.txt" "$model" 2>&1)
    assert_match "predict produce PREDICTIONS"       "PREDICTIONS"          "$out_predict"
    assert_match "predict produce EVALUATION METRICS" "EVALUATION METRICS"  "$out_predict"
    assert_match "predict produce MSE"               "Mean Squared Error"   "$out_predict"
    assert_match "predict produce Accuracy"          "Accuracy"             "$out_predict"
}

_test_predict_separate

suite_begin "Pipeline — eval = train + predict in sequenza"

_test_eval_sequence() {
    local model out
    model=$(make_tmp_model "pipe_eval" 2,4,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" eval "$ROOT_DIR/dataset/xor.txt" "$model" \
        --epochs 1000 --optimizer adam 2>&1)
    # eval stampa PREDICTIONS due volte (una da train con print_result, una da predict)
    local count
    count=$(echo "$out" | grep -c "PREDICTIONS")
    if [[ "$count" -ge 2 ]]; then
        _pass "eval mostra PREDICTIONS almeno 2 volte (train + predict)"
    else
        _fail "eval mostra PREDICTIONS almeno 2 volte" "trovate $count occorrenze"
    fi
}

_test_eval_sequence

cleanup_tmp

suite_summary
