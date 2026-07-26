#!/usr/bin/gawk -f
#
# Test unitari AWK per le funzioni di loss.
#

function nearly_eq(a, b, tol,    diff) {
    tol  = (tol == "" ? 1e-6 : tol)
    diff = (a > b ? a - b : b - a)
    return (diff <= tol)
}

function check(name, got, expected, tol) {
    if (nearly_eq(got, expected, tol)) {
        print "PASS " name
    } else {
        printf "FAIL %s  got=%.10g  expected=%.10g\n", name, got, expected
    }
}

BEGIN {
    # ----------------------------------------------------------------
    # compute_sample_loss — MSE
    # ----------------------------------------------------------------
    # MSE = 0.5 * (output - target)^2
    check("mse: output=target → 0",
        compute_sample_loss(0.5, 0.5, "sigmoid", "mse"), 0.0)

    check("mse: output=0 target=1 → 0.5",
        compute_sample_loss(0.0, 1.0, "sigmoid", "mse"), 0.5)

    check("mse: output=1 target=0 → 0.5",
        compute_sample_loss(1.0, 0.0, "sigmoid", "mse"), 0.5)

    check("mse: output=0.8 target=0.2 → 0.18",
        compute_sample_loss(0.8, 0.2, "sigmoid", "mse"), 0.18)

    # ----------------------------------------------------------------
    # compute_sample_loss — CE (solo con sigmoid)
    # ----------------------------------------------------------------
    # CE = -[t*log(y) + (1-t)*log(1-y)]
    # CE(y=1, t=1) = -log(1) = 0
    check("ce sigmoid: output~1 target=1 → ~0",
        compute_sample_loss(0.9999999, 1.0, "sigmoid", "ce"), 0.0, 1e-5)

    # CE(y=0, t=0) = -log(1) = 0
    check("ce sigmoid: output~0 target=0 → ~0",
        compute_sample_loss(1e-10, 0.0, "sigmoid", "ce"), 0.0, 1e-5)

    # CE(y=0.5, t=1) = -log(0.5) = log(2) ≈ 0.6931
    check("ce sigmoid: output=0.5 target=1 → log(2)",
        compute_sample_loss(0.5, 1.0, "sigmoid", "ce"), log(2), 1e-6)

    # CE non-sigmoid deve fare fallback a MSE
    check("ce tanh: fallback a mse",
        compute_sample_loss(0.0, 1.0, "tanh", "ce"),
        compute_sample_loss(0.0, 1.0, "tanh", "mse"))

    # ----------------------------------------------------------------
    # compute_output_delta — MSE
    # ----------------------------------------------------------------
    # delta = (output - target) * d_sigmoid(output)
    # output=0.5: error=0.5-1.0=-0.5, d_sigmoid(0.5)=0.25 → delta=-0.125
    check("delta mse sigmoid output=0.5 target=1",
        compute_output_delta(0.0, 0.5, 1.0, "sigmoid", "mse", 0.01), -0.125)

    # ----------------------------------------------------------------
    # compute_output_delta — CE + sigmoid (semplificazione a output-target)
    # ----------------------------------------------------------------
    # delta CE+sigmoid = output - target
    check("delta ce sigmoid output=0.7 target=1 → -0.3",
        compute_output_delta(0.0, 0.7, 1.0, "sigmoid", "ce", 0.01), -0.3)

    check("delta ce sigmoid output=0.2 target=0 → 0.2",
        compute_output_delta(0.0, 0.2, 0.0, "sigmoid", "ce", 0.01), 0.2)
}
