#!/bin/bash
#
# Dispatcher: test di inizializzazione rete.
# Verifica che nnet-init.sh crei file corretti per tutte le combinazioni
# di attivazione, metodo e topologia.
#

source "$(dirname "$0")/lib/framework.sh"

suite_begin "Init — file creati"

_test_init_creates_files() {
    local name="$1"; shift
    local model
    model=$(make_tmp_model "$name" "$@")
    assert_file_exists "$name: layer1.txt" "$model/layer1.txt"
    assert_file_exists "$name: layer2.txt" "$model/layer2.txt"
    assert_file_exists "$name: model.conf" "$model/model.conf"
}

_test_init_creates_files "sigmoid/xavier/2,3,1"   2,3,1  --activation sigmoid --method xavier
_test_init_creates_files "tanh/he/2,4,1"          2,4,1  --activation tanh    --method he
_test_init_creates_files "relu/he/2,4,1"          2,4,1  --activation relu    --method he
_test_init_creates_files "leaky_relu/2,4,1"       2,4,1  --activation leaky_relu --alpha 0.1
_test_init_creates_files "random/2,3,1"           2,3,1  --activation sigmoid --method random
_test_init_creates_files "deep/2,8,4,1"           2,8,4,1 --activation sigmoid

suite_begin "Init — header ACTIVATION nei file layer"

_test_activation_header() {
    local name="$1" arch="$2" expected_header="$3"; shift 3
    local model
    model=$(make_tmp_model "$name" "$arch" "$@")
    assert_file_contains "$name: header layer1" "$model/layer1.txt" "^ACTIVATION=$expected_header"
    assert_file_contains "$name: header layer2" "$model/layer2.txt" "^ACTIVATION=$expected_header"
}

_test_activation_header "sigmoid header"          2,3,1 "sigmoid"        --activation sigmoid
_test_activation_header "tanh header"             2,3,1 "tanh"           --activation tanh
_test_activation_header "relu header"             2,3,1 "relu"           --activation relu
_test_activation_header "leaky_relu:0.1 header"   2,3,1 "leaky_relu:0.1" --activation leaky_relu --alpha 0.1
_test_activation_header "leaky_relu:0.2 header"   2,3,1 "leaky_relu:0.2" --activation leaky_relu --alpha 0.2

suite_begin "Init — numero pesi corretto"

_test_weight_count() {
    local name="$1" arch="$2" layer="$3" expected_cols="$4"; shift 4
    local model
    model=$(make_tmp_model "$name" "$arch" "$@")
    local file="$model/layer${layer}.txt"
    # Conta colonne sulla seconda riga (prima riga è ACTIVATION=)
    local ncols
    ncols=$(awk 'NR==2{print NF; exit}' "$file")
    assert_eq "$name: layer${layer} ha $expected_cols colonne" "$expected_cols" "$ncols"
}

# 2,3,1 → layer1: 3 neuroni × (2 input + 1 bias) = 3 colonne per riga
#          layer2: 1 neurone  × (3 input + 1 bias) = 4 colonne per riga
_test_weight_count "2,3,1 layer1 cols" 2,3,1 1 3 --activation sigmoid
_test_weight_count "2,3,1 layer2 cols" 2,3,1 2 4 --activation sigmoid

# 2,4,2 → layer1: 4 neuroni × 3 = 3 col; layer2: 2 neuroni × 5 = 5 col
_test_weight_count "2,4,2 layer1 cols" 2,4,2 1 3 --activation sigmoid
_test_weight_count "2,4,2 layer2 cols" 2,4,2 2 5 --activation sigmoid

suite_begin "Init — validazione input errati"

assert_exit_fail "activation invalida"  "$NNET_INIT" "$TMPDIR_TESTS/bad1" 2,3,1 --activation invalid_fn
assert_exit_fail "method invalido"      "$NNET_INIT" "$TMPDIR_TESTS/bad2" 2,3,1 --method unknown
assert_exit_fail "architettura vuota"   "$NNET_INIT" "$TMPDIR_TESTS/bad3" ""
assert_exit_fail "singolo layer"        "$NNET_INIT" "$TMPDIR_TESTS/bad4" 3

cleanup_tmp

suite_summary
