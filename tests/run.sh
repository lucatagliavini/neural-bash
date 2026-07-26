#!/bin/bash
#
# Orchestratore della test suite di neural-bash.
#
# Usage:
#   bash tests/run.sh              # esegue tutti i dispatcher
#   bash tests/run.sh unit         # solo test unitari
#   bash tests/run.sh init         # solo test di inizializzazione
#   bash tests/run.sh activations  # solo test attivazioni
#   bash tests/run.sh optimizers   # solo test optimizer
#   bash tests/run.sh loss         # solo test loss
#   bash tests/run.sh checkpoint   # solo test checkpoint
#   bash tests/run.sh pipeline     # solo test end-to-end
#

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colori (ridefiniti qui per l'output dell'orchestratore)
if [[ -t 1 ]]; then
    _GREEN="\033[0;32m"; _RED="\033[0;31m"; _YELLOW="\033[0;33m"
    _BOLD="\033[1m"; _RESET="\033[0m"
else
    _GREEN="" _RED="" _YELLOW="" _BOLD="" _RESET=""
fi

# ============================================================================
# Registro dei dispatcher
# Aggiungere una riga qui quando si crea un nuovo dispatcher.
# ============================================================================

declare -A DISPATCHERS=(
    [unit]="$TESTS_DIR/test_unit.sh"
    [init]="$TESTS_DIR/test_init.sh"
    [activations]="$TESTS_DIR/test_activations.sh"
    [optimizers]="$TESTS_DIR/test_optimizers.sh"
    [loss]="$TESTS_DIR/test_loss.sh"
    [checkpoint]="$TESTS_DIR/test_checkpoint.sh"
    [pipeline]="$TESTS_DIR/test_pipeline.sh"
    [val_split]="$TESTS_DIR/test_val_split.sh"
    [multiclass]="$TESTS_DIR/test_multiclass.sh"
    [metrics]="$TESTS_DIR/test_metrics.sh"
    [normalize]="$TESTS_DIR/test_normalize.sh"
    [dropout]="$TESTS_DIR/test_dropout.sh"
    [batch]="$TESTS_DIR/test_batch.sh"
    [search]="$TESTS_DIR/test_search.sh"
)

DISPATCHER_ORDER=(unit init activations optimizers loss checkpoint pipeline val_split multiclass metrics normalize dropout batch search)

# ============================================================================
# Selezione dispatcher
# ============================================================================

if [[ $# -gt 0 ]]; then
    if [[ -z "${DISPATCHERS[$1]}" ]]; then
        echo "Dispatcher sconosciuto: $1"
        echo "Disponibili: ${DISPATCHER_ORDER[*]}"
        exit 1
    fi
    SELECTED=("$1")
else
    SELECTED=("${DISPATCHER_ORDER[@]}")
fi

# ============================================================================
# Esecuzione
# ============================================================================

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0

echo ""
echo -e "${_BOLD}╔══════════════════════════════════════════════════════╗${_RESET}"
echo -e "${_BOLD}║          neural-bash — Test Suite                   ║${_RESET}"
echo -e "${_BOLD}╚══════════════════════════════════════════════════════╝${_RESET}"

for name in "${SELECTED[@]}"; do
    script="${DISPATCHERS[$name]}"

    if [[ ! -f "$script" ]]; then
        echo -e "\n${_YELLOW}[SKIP]${_RESET} dispatcher '$name' non trovato: $script"
        continue
    fi

    chmod +x "$script"

    # Ogni dispatcher gira in subshell: i suoi contatori _PASS/_FAIL/_SKIP
    # vengono stampati via suite_summary e catturati qui.
    output=$(bash "$script" 2>&1)
    echo "$output"

    # Estrae i totali dal sommario stampato da suite_summary
    pass=$(echo "$output" | grep -oE "[0-9]+ passed"  | grep -oE "^[0-9]+")
    fail=$(echo "$output" | grep -oE "[0-9]+ failed"  | grep -oE "^[0-9]+")
    skip=$(echo "$output" | grep -oE "[0-9]+ skipped" | grep -oE "^[0-9]+")

    TOTAL_PASS=$((TOTAL_PASS + ${pass:-0}))
    TOTAL_FAIL=$((TOTAL_FAIL + ${fail:-0}))
    TOTAL_SKIP=$((TOTAL_SKIP + ${skip:-0}))
done

# ============================================================================
# Sommario globale
# ============================================================================

TOTAL=$((TOTAL_PASS + TOTAL_FAIL + TOTAL_SKIP))

echo ""
echo -e "${_BOLD}══════════════════════════════════════════════════════${_RESET}"
echo -e "${_BOLD}TOTALE: ${_GREEN}$TOTAL_PASS passed${_RESET}${_BOLD}, ${_RED}$TOTAL_FAIL failed${_RESET}${_BOLD}, ${_YELLOW}$TOTAL_SKIP skipped${_RESET}${_BOLD} / $TOTAL${_RESET}"
echo -e "${_BOLD}══════════════════════════════════════════════════════${_RESET}"
echo ""

# Exit code 0 solo se nessun fail
[[ $TOTAL_FAIL -eq 0 ]]
