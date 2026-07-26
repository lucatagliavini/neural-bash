#!/bin/bash
#
# Libreria condivisa per la test suite di neural-bash.
# Caricata da ogni dispatcher con: source "$(dirname "$0")/lib/framework.sh"
#

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd "$TESTS_DIR/.." && pwd)"
NNET_RUN="$ROOT_DIR/nnet-run.sh"
NNET_INIT="$ROOT_DIR/nnet-init.sh"
AWK_LIB="$ROOT_DIR/lib/framework"
TMPDIR_TESTS="$TESTS_DIR/tmp"

# Contatori globali (accumulati da ogni dispatcher nell'orchestratore)
_PASS=0
_FAIL=0
_SKIP=0
_CURRENT_SUITE=""

# ============================================================================
# COLORI
# ============================================================================

if [[ -t 1 ]]; then
    _GREEN="\033[0;32m"
    _RED="\033[0;31m"
    _YELLOW="\033[0;33m"
    _CYAN="\033[0;36m"
    _BOLD="\033[1m"
    _RESET="\033[0m"
else
    _GREEN="" _RED="" _YELLOW="" _CYAN="" _BOLD="" _RESET=""
fi

# ============================================================================
# REPORTING
# ============================================================================

suite_begin() {
    _CURRENT_SUITE="$1"
    echo ""
    echo -e "${_BOLD}${_CYAN}━━━ $1 ━━━${_RESET}"
}

_pass() {
    local name="$1"
    _PASS=$((_PASS + 1))
    echo -e "  ${_GREEN}[PASS]${_RESET} $name"
}

_fail() {
    local name="$1"
    local reason="$2"
    _FAIL=$((_FAIL + 1))
    echo -e "  ${_RED}[FAIL]${_RESET} $name"
    echo -e "         ${_RED}↳ $reason${_RESET}"
}

_skip() {
    local name="$1"
    local reason="$2"
    _SKIP=$((_SKIP + 1))
    echo -e "  ${_YELLOW}[SKIP]${_RESET} $name — $reason"
}

suite_summary() {
    local total=$((_PASS + _FAIL + _SKIP))
    echo ""
    echo -e "${_BOLD}Risultato: ${_GREEN}$_PASS passed${_RESET}${_BOLD}, ${_RED}$_FAIL failed${_RESET}${_BOLD}, ${_YELLOW}$_SKIP skipped${_RESET}${_BOLD} / $total total${_RESET}"
}

# ============================================================================
# ASSERZIONI PRIMITIVE
# ============================================================================

# assert_eq <nome> <atteso> <ottenuto>
assert_eq() {
    local name="$1" expected="$2" got="$3"
    if [[ "$expected" == "$got" ]]; then
        _pass "$name"
    else
        _fail "$name" "atteso='$expected' ottenuto='$got'"
    fi
}

# assert_match <nome> <pattern_regex> <stringa>
assert_match() {
    local name="$1" pattern="$2" value="$3"
    if echo "$value" | grep -qE "$pattern"; then
        _pass "$name"
    else
        _fail "$name" "pattern='$pattern' non trovato in '$value'"
    fi
}

# assert_file_contains <nome> <file> <pattern_regex>
assert_file_contains() {
    local name="$1" file="$2" pattern="$3"
    if [[ ! -f "$file" ]]; then
        _fail "$name" "file non trovato: $file"
        return
    fi
    if grep -qE "$pattern" "$file"; then
        _pass "$name"
    else
        _fail "$name" "pattern='$pattern' non trovato in $file"
    fi
}

# assert_file_exists <nome> <file>
assert_file_exists() {
    local name="$1" file="$2"
    if [[ -f "$file" ]]; then
        _pass "$name"
    else
        _fail "$name" "file non trovato: $file"
    fi
}

# assert_exit_ok <nome> <comando...>
assert_exit_ok() {
    local name="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        _pass "$name"
    else
        _fail "$name" "comando uscito con codice non-zero: $*"
    fi
}

# assert_exit_fail <nome> <comando...>
assert_exit_fail() {
    local name="$1"
    shift
    if ! "$@" > /dev/null 2>&1; then
        _pass "$name"
    else
        _fail "$name" "comando atteso in errore ma uscito con 0: $*"
    fi
}

# assert_mse_below <nome> <soglia> <output_del_comando>
# Estrae MSE dall'output di nnet-run.sh e verifica che sia sotto la soglia.
assert_mse_below() {
    local name="$1" threshold="$2" output="$3"
    local mse
    mse=$(echo "$output" | grep -oE "Mean Squared Error \(MSE\) +: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    if [[ -z "$mse" ]]; then
        _fail "$name" "MSE non trovato nell'output"
        return
    fi
    if awk "BEGIN { exit ($mse < $threshold) ? 0 : 1 }"; then
        _pass "$name (MSE=$mse < $threshold)"
    else
        _fail "$name" "MSE=$mse >= soglia=$threshold"
    fi
}

# assert_accuracy_above <nome> <soglia_pct> <output_del_comando>
# Estrae Accuracy% dall'output e verifica che sia sopra la soglia.
assert_accuracy_above() {
    local name="$1" threshold="$2" output="$3"
    local acc
    acc=$(echo "$output" | grep -oE "Accuracy +: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    if [[ -z "$acc" ]]; then
        _fail "$name" "Accuracy non trovata nell'output"
        return
    fi
    if awk "BEGIN { exit ($acc > $threshold) ? 0 : 1 }"; then
        _pass "$name (Accuracy=${acc}% > ${threshold}%)"
    else
        _fail "$name" "Accuracy=${acc}% <= soglia=${threshold}%"
    fi
}

# ============================================================================
# HELPERS
# ============================================================================

# Crea un modello temporaneo e restituisce il path. Pulito da cleanup_tmp.
make_tmp_model() {
    local name="$1"; shift   # es. "xor_sgd"
    local dir="$TMPDIR_TESTS/model_$name"
    "$NNET_INIT" "$dir" "$@" --force > /dev/null 2>&1
    echo "$dir"
}

# Rimuove tutti i modelli temporanei creati durante i test.
cleanup_tmp() {
    rm -rf "$TMPDIR_TESTS"/model_*
}

# Esegue un forward AWK puro per test unitari di singole funzioni.
# Uso: awk_eval <espressione_awk>  →  restituisce il valore su stdout
awk_eval() {
    awk -f "$AWK_LIB/utils-math.awk" \
        -f "$AWK_LIB/utils-activation.awk" \
        -f "$AWK_LIB/utils-loss.awk" \
        -f "$AWK_LIB/utils-shared.awk" \
        -v expr="$1" \
        'BEGIN { printf "%.10g\n", eval(expr) }' \
        /dev/null 2>/dev/null
}

# Esegue AWK con tutti i moduli e un programma inline.
# Uso: awk_run <programma_awk>
awk_run() {
    local prog="$1"
    awk -f "$AWK_LIB/utils-math.awk" \
        -f "$AWK_LIB/utils-activation.awk" \
        -f "$AWK_LIB/utils-loss.awk" \
        -f "$AWK_LIB/utils-shared.awk" \
        "BEGIN { $prog }" \
        /dev/null 2>/dev/null
}
