#!/bin/bash
#
# Dispatcher: test confusion matrix e task=regression/classification.
#

source "$(dirname "$0")/lib/framework.sh"

XOR_DS="$ROOT_DIR/dataset/xor.txt"
IRIS_DS="$ROOT_DIR/dataset/iris_toy.txt"
SINE_DS="$ROOT_DIR/dataset/sine_regression.txt"

# ---------------------------------------------------------------------------
suite_begin "Metrics — MAE presente nell'output di classificazione"

_test_mae_classification() {
    local model out
    model=$(make_tmp_model "met_mae_cls" 2,4,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$XOR_DS" "$model" --epochs 500 --optimizer adam 2>&1)
    assert_match "MAE presente in output classificazione" "Mean Absolute Error" "$out"
}

_test_mae_classification

# ---------------------------------------------------------------------------
suite_begin "Metrics — confusion matrix 2x2 su classificazione binaria"

_test_cm_binary() {
    local model out
    model=$(make_tmp_model "met_cm2" 2,4,1 --activation sigmoid --seed 42)
    out=$("$NNET_RUN" train "$XOR_DS" "$model" --epochs 1000 --optimizer adam 2>&1)
    assert_match "confusion matrix presente"    "CONFUSION MATRIX"  "$out"
    assert_match "riga C1 presente nella CM"    "^C1"               "$out"
    assert_match "riga C2 presente nella CM"    "^C2"               "$out"
}

_test_cm_binary

# ---------------------------------------------------------------------------
suite_begin "Metrics — confusion matrix 3x3 su classificazione multi-classe"

_test_cm_multiclass() {
    local model out
    model=$(make_tmp_model "met_cm3" 4,8,3 \
        --activation softmax --hidden-act sigmoid --method xavier --seed 7)
    out=$("$NNET_RUN" train "$IRIS_DS" "$model" \
        --epochs 3000 --optimizer adam --loss cce --lr 0.01 2>&1)
    assert_match "confusion matrix presente"   "CONFUSION MATRIX"  "$out"
    assert_match "riga C3 presente nella CM"   "^C3"               "$out"
    assert_accuracy_above "iris accuracy > 90%" 90 "$out"
}

_test_cm_multiclass

# ---------------------------------------------------------------------------
suite_begin "Metrics — confusion matrix diagonale piena su iris convergente"

_test_cm_diagonal() {
    local model out
    model=$(make_tmp_model "met_diag" 4,8,3 \
        --activation softmax --hidden-act sigmoid --method xavier --seed 7)
    "$NNET_RUN" train "$IRIS_DS" "$model" \
        --epochs 3000 --optimizer adam --loss cce --lr 0.01 > /dev/null 2>&1
    out=$("$NNET_RUN" predict "$IRIS_DS" "$model" 2>&1)
    # Con 100% accuracy la diagonale deve avere tutti valori > 0 e gli off-diagonal = 0
    local acc
    acc=$(echo "$out" | grep -oE "Accuracy +: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    if [[ -n "$acc" ]] && awk -v a="$acc" 'BEGIN { exit (a >= 99) ? 0 : 1 }'; then
        # Conta gli zeri fuori diagonale: C1..C2, C1..C3, C2..C1, ecc.
        # Verifica assenza di errori cercando "0" nelle righe C1/C2/C3 dopo il primo campo
        local ok=1
        while IFS= read -r line; do
            if echo "$line" | grep -qE "^C[0-9]"; then
                # valori numerici nella riga (escluso il label)
                local vals
                vals=$(echo "$line" | awk '{for(i=2;i<=NF;i++) printf $i " "; print ""}')
                # conta quanti valori sono 0 — devono essere esattamente n_classes-1
                local zeros total
                zeros=$(echo "$vals" | tr ' ' '\n' | grep -c '^0$' || true)
                total=$(echo "$vals" | tr ' ' '\n' | grep -c '[0-9]' || true)
                if [[ $((total - zeros)) -ne 1 ]]; then ok=0; fi
            fi
        done <<< "$out"
        if [[ $ok -eq 1 ]]; then
            _pass "CM diagonale pura (nessun errore di classificazione)"
        else
            _fail "CM diagonale pura" "trovati valori off-diagonal non-zero"
        fi
    else
        _skip "CM diagonale (accuracy=$acc% < 99%, skip check diagonale)" "accuracy insufficiente"
    fi
}

_test_cm_diagonal

# ---------------------------------------------------------------------------
suite_begin "Metrics — task=regression: nessuna confusion matrix né Accuracy"

_test_regression_no_cm() {
    local model out
    model=$(make_tmp_model "met_reg_nocm" 1,16,16,1 \
        --activation linear --hidden-act tanh --method xavier --seed 5)
    out=$("$NNET_RUN" train "$SINE_DS" "$model" \
        --epochs 1000 --optimizer adam --lr 0.005 --task regression 2>&1)
    if echo "$out" | grep -q "CONFUSION MATRIX"; then
        _fail "regressione: nessuna CM" "trovata CONFUSION MATRIX nell'output"
    else
        _pass "regressione: nessuna CONFUSION MATRIX"
    fi
    if echo "$out" | grep -q "^Accuracy"; then
        _fail "regressione: nessuna Accuracy" "trovata riga Accuracy nell'output"
    else
        _pass "regressione: nessuna riga Accuracy"
    fi
}

_test_regression_no_cm

# ---------------------------------------------------------------------------
suite_begin "Metrics — task=regression: MAE e MSE presenti"

_test_regression_mae_mse() {
    local model out
    model=$(make_tmp_model "met_reg_mae" 1,16,16,1 \
        --activation linear --hidden-act tanh --method xavier --seed 5)
    out=$("$NNET_RUN" train "$SINE_DS" "$model" \
        --epochs 1000 --optimizer adam --lr 0.005 --task regression 2>&1)
    assert_match "MAE presente in output regressione" "Mean Absolute Error" "$out"
    assert_match "MSE presente in output regressione" "Mean Squared Error"  "$out"
}

_test_regression_mae_mse

# ---------------------------------------------------------------------------
suite_begin "Metrics — regressione seno converge (MSE < 0.01)"

_test_regression_converges() {
    local model out
    model=$(make_tmp_model "met_reg_conv" 1,16,16,1 \
        --activation linear --hidden-act tanh --method xavier --seed 5)
    out=$("$NNET_RUN" train "$SINE_DS" "$model" \
        --epochs 3000 --optimizer adam --lr 0.005 --task regression 2>&1)
    assert_mse_below "regressione seno MSE < 0.01" 0.01 "$out"
}

_test_regression_converges

# ---------------------------------------------------------------------------
suite_begin "Metrics — predict regressione mostra Abs.Error per campione"

_test_regression_predict_format() {
    local model out
    model=$(make_tmp_model "met_reg_fmt" 1,16,16,1 \
        --activation linear --hidden-act tanh --method xavier --seed 5)
    "$NNET_RUN" train "$SINE_DS" "$model" \
        --epochs 2000 --optimizer adam --lr 0.005 --task regression > /dev/null 2>&1
    out=$("$NNET_RUN" predict "$SINE_DS" "$model" --task regression 2>&1)
    assert_match "predict regression mostra Abs.Error" "Abs.Error" "$out"
    assert_match "predict regression mostra PREDICTIONS" "PREDICTIONS" "$out"
}

_test_regression_predict_format

cleanup_tmp

suite_summary
