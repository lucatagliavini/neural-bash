#!/bin/bash
#
# Dispatcher: test unitari AWK (funzioni matematiche, attivazioni, loss).
# Esegue ogni file .awk in tests/unit/ e converte PASS/FAIL in assert del framework.
#

source "$(dirname "$0")/lib/framework.sh"

_run_awk_unit() {
    local label="$1" awk_file="$2"
    local output
    output=$(gawk \
        -f "$AWK_LIB/utils-math.awk" \
        -f "$AWK_LIB/utils-activation.awk" \
        -f "$AWK_LIB/utils-loss.awk" \
        -f "$AWK_LIB/utils-shared.awk" \
        -f "$awk_file" \
        /dev/null 2>/dev/null)

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local status="${line%% *}"
        local name="${line#* }"
        if [[ "$status" == "PASS" ]]; then
            _pass "$name"
        else
            _fail "${name%% *}" "${name#* }"
        fi
    done <<< "$output"
}

suite_begin "Unit — utils-math"
_run_awk_unit "math" "$TESTS_DIR/unit/test_math.awk"

suite_begin "Unit — utils-activation"
_run_awk_unit "activation" "$TESTS_DIR/unit/test_activations.awk"

suite_begin "Unit — utils-loss"
_run_awk_unit "loss" "$TESTS_DIR/unit/test_loss.awk"

suite_summary
