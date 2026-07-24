#!/usr/bin/awk -f
#
# Test unitari per softmax (apply_softmax) e compute_output_delta con CCE.
#

function nearly_eq(a, b, tol) { return (a - b < tol && b - a < tol) }

function check(name, got, expected, tol,    ok) {
    if (tol == "") tol = 1e-5
    ok = nearly_eq(got, expected, tol)
    if (ok) {
        printf "PASS %s\n", name
        PASS++
    } else {
        printf "FAIL %s: got=%.8g expected=%.8g\n", name, got, expected
        FAIL++
    }
}

BEGIN {
    PASS = 0; FAIL = 0

    # ---------------------------------------------------------------------------
    # Test apply_softmax
    # ---------------------------------------------------------------------------
    lo[1,1,1] = 1; lo[1,1,2] = 2; lo[1,1,3] = 3
    apply_softmax(lo, 1, 1, 3)
    s = exp(1) + exp(2) + exp(3)
    check("softmax([1,2,3])[1]",    lo[1,1,1], exp(1)/s)
    check("softmax([1,2,3])[2]",    lo[1,1,2], exp(2)/s)
    check("softmax([1,2,3])[3]",    lo[1,1,3], exp(3)/s)

    # somma = 1
    total = lo[1,1,1] + lo[1,1,2] + lo[1,1,3]
    check("softmax somma = 1",       total, 1.0, 1e-9)

    # invarianza alla traslazione (stabilità numerica)
    lo2[1,1,1] = 1001; lo2[1,1,2] = 1002; lo2[1,1,3] = 1003
    apply_softmax(lo2, 1, 1, 3)
    check("softmax stabilità [1]",   lo2[1,1,1], exp(1)/s, 1e-6)
    check("softmax stabilità [2]",   lo2[1,1,2], exp(2)/s, 1e-6)
    check("softmax stabilità [3]",   lo2[1,1,3], exp(3)/s, 1e-6)

    # distribuzione uniforme
    lo3[1,1,1] = 5; lo3[1,1,2] = 5; lo3[1,1,3] = 5
    apply_softmax(lo3, 1, 1, 3)
    check("softmax uniforme [1]",    lo3[1,1,1], 1/3, 1e-9)
    check("softmax uniforme [2]",    lo3[1,1,2], 1/3, 1e-9)
    check("softmax uniforme [3]",    lo3[1,1,3], 1/3, 1e-9)

    # ---------------------------------------------------------------------------
    # Test compute_output_delta con CCE+softmax: delta = output - target
    # ---------------------------------------------------------------------------
    check("delta cce+softmax p=0.8 t=1",   compute_output_delta(0.8, 1.0, "softmax", "cce"), -0.2)
    check("delta cce+softmax p=0.1 t=0",   compute_output_delta(0.1, 0.0, "softmax", "cce"),  0.1)
    check("delta cce+softmax p=0.5 t=0.5", compute_output_delta(0.5, 0.5, "softmax", "cce"),  0.0)

    # ---------------------------------------------------------------------------
    # Test compute_sample_loss con CCE
    # ---------------------------------------------------------------------------
    check("cce loss: p~1 t=1 → ~0",   compute_sample_loss(1.0-1e-15, 1.0, "softmax", "cce"), 0.0, 1e-4)
    check("cce loss: p=0.5 t=1",       compute_sample_loss(0.5,       1.0, "softmax", "cce"), log(2), 1e-6)
    check("cce loss: p=0.1 t=0 → 0",  compute_sample_loss(0.1,       0.0, "softmax", "cce"), 0.0)

    exit (FAIL > 0 ? 1 : 0)
}
