#!/bin/bash
#
# Dispatcher: test nnet-search.sh (hyperparameter search).
#

source "$(dirname "$0")/lib/framework.sh"

DATASET_XOR="$ROOT_DIR/dataset/xor.txt"
NNET_SEARCH="$ROOT_DIR/nnet-search.sh"
SEARCH_PREFIX="$TMPDIR_TESTS/search"

# ---------------------------------------------------------------------------
suite_begin "nnet-search — help e validazione input"

_test_search_help() {
    local out
    out=$("$NNET_SEARCH" --help 2>&1 || true)
    assert_match "help mostra Usage" "Usage:" "$out"
}

_test_search_missing_arch() {
    local out
    out=$("$NNET_SEARCH" "$DATASET_XOR" "${SEARCH_PREFIX}_noarch" 2>&1 || true)
    assert_match "--arch mancante dà errore" "arch.*obbligatorio" "$out"
}

_test_search_missing_dataset() {
    local out
    out=$("$NNET_SEARCH" /nonexistent/path.txt "${SEARCH_PREFIX}_nods" \
        --arch "2,4,1" 2>&1 || true)
    assert_match "dataset mancante dà errore" "Dataset non trovato" "$out"
}

_test_search_help
_test_search_missing_arch
_test_search_missing_dataset

# ---------------------------------------------------------------------------
suite_begin "nnet-search — output tabella ranking presente"

_test_search_table() {
    local out
    out=$("$NNET_SEARCH" "$DATASET_XOR" "${SEARCH_PREFIX}_table" \
        --arch "2,4,1" \
        --lr "0.1" \
        --optimizer "sgd" \
        --epochs 100 \
        --jobs 1 2>&1)
    assert_match "header RANK presente"      "RANK"       "$out"
    assert_match "header BEST_MSE presente"  "BEST_MSE"   "$out"
    assert_match "header OPTIMIZER presente" "OPTIMIZER"  "$out"
    assert_match "riga risultato presente"   "sgd"        "$out"
}

_test_search_table

# ---------------------------------------------------------------------------
suite_begin "nnet-search — smoke test: 4 combinazioni (2 arch x 2 lr)"

_test_search_4combos() {
    local out
    out=$("$NNET_SEARCH" "$DATASET_XOR" "${SEARCH_PREFIX}_4c" \
        --arch "2,4,1 2,8,1" \
        --lr "0.01 0.1" \
        --optimizer "sgd" \
        --epochs 200 \
        --jobs 2 2>&1)
    # Devono esserci 4 righe di risultati (rank 1..4)
    local count
    count=$(echo "$out" | grep -cE "^\s*[0-9]+\s+[0-9,]+\s" || true)
    if [[ "$count" -eq 4 ]]; then
        _pass "4 combinazioni: 4 righe nel ranking"
    else
        _fail "4 combinazioni: 4 righe nel ranking" "trovate $count righe"
    fi
}

_test_search_4combos

# ---------------------------------------------------------------------------
suite_begin "nnet-search — best_mse nel ranking è un numero"

_test_search_mse_numeric() {
    local out
    out=$("$NNET_SEARCH" "$DATASET_XOR" "${SEARCH_PREFIX}_mse" \
        --arch "2,4,1" \
        --lr "0.1" \
        --optimizer "adam" \
        --epochs 500 \
        --jobs 1 2>&1)
    # Estrae il valore best_mse dalla prima riga di risultato
    local mse
    mse=$(echo "$out" | grep -E "^\s*1\s" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [[ -n "$mse" ]]; then
        _pass "best_mse è numerico ($mse)"
    else
        _fail "best_mse è numerico" "valore non trovato nell'output: $out"
    fi
}

_test_search_mse_numeric

# ---------------------------------------------------------------------------
suite_begin "nnet-search — rank 1 ha best_mse <= rank 2 (ordinamento corretto)"

_test_search_sorted() {
    local out
    out=$("$NNET_SEARCH" "$DATASET_XOR" "${SEARCH_PREFIX}_sort" \
        --arch "2,4,1 2,8,1" \
        --lr "0.01 0.1" \
        --optimizer "adam" \
        --epochs 500 \
        --jobs 2 2>&1)
    local mse1 mse2
    mse1=$(echo "$out" | grep -E "^\s*1\s" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    mse2=$(echo "$out" | grep -E "^\s*2\s" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [[ -z "$mse1" || -z "$mse2" ]]; then
        _fail "ordinamento: due righe nel ranking" "mse1='$mse1' mse2='$mse2'"
        return
    fi
    if awk "BEGIN { exit ($mse1 <= $mse2) ? 0 : 1 }"; then
        _pass "rank1 MSE ($mse1) <= rank2 MSE ($mse2)"
    else
        _fail "rank1 MSE ($mse1) <= rank2 MSE ($mse2)" "ordinamento non corretto"
    fi
}

_test_search_sorted

# ---------------------------------------------------------------------------
suite_begin "nnet-search — cleanup: nessuna directory temporanea rimasta"

_test_search_cleanup() {
    "$NNET_SEARCH" "$DATASET_XOR" "${SEARCH_PREFIX}_clean" \
        --arch "2,4,1" \
        --lr "0.1" \
        --optimizer "sgd" \
        --epochs 50 \
        --jobs 1 > /dev/null 2>&1
    # La directory _search_<PID> deve essere stata rimossa
    local leftovers
    leftovers=$(find "$TMPDIR_TESTS" -maxdepth 1 -name "search_clean_search_*" -type d 2>/dev/null | wc -l)
    if [[ "$leftovers" -eq 0 ]]; then
        _pass "nessuna directory temporanea rimasta"
    else
        _fail "nessuna directory temporanea rimasta" "$leftovers directory trovate"
    fi
}

_test_search_cleanup

cleanup_tmp
suite_summary
