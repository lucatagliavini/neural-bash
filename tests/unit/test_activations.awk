#!/usr/bin/gawk -f
#
# Test unitari AWK per le funzioni di attivazione.
# Eseguito da test_activations.sh tramite gawk -f.
#
# Ogni test stampa: PASS <nome> oppure FAIL <nome> <motivo>
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
    # sigmoid
    # ----------------------------------------------------------------
    check("sigmoid(0)=0.5",           f_sigmoid(0),      0.5)
    check("sigmoid(large)~1",         f_sigmoid(100),    1.0,    1e-6)
    check("sigmoid(-large)~0",        f_sigmoid(-100),   0.0,    1e-6)
    check("sigmoid(1)~0.731",         f_sigmoid(1),      0.7310585786, 1e-6)

    # ----------------------------------------------------------------
    # d_sigmoid (lavora sull'output già attivato)
    # ----------------------------------------------------------------
    check("d_sigmoid(0.5)=0.25",      d_sigmoid(0.5),    0.25)
    check("d_sigmoid(0)=0",           d_sigmoid(0),      0.0)
    check("d_sigmoid(1)=0",           d_sigmoid(1),      0.0)

    # ----------------------------------------------------------------
    # tanh
    # ----------------------------------------------------------------
    check("tanh(0)=0",                f_tanh(0),         0.0)
    check("tanh(large)~1",            f_tanh(100),       1.0,    1e-6)
    check("tanh(-large)~-1",          f_tanh(-100),     -1.0,    1e-6)
    check("tanh(1)~0.7616",           f_tanh(1),         0.7615941559, 1e-6)

    # ----------------------------------------------------------------
    # d_tanh (lavora sull'output già attivato)
    # ----------------------------------------------------------------
    check("d_tanh(0)=1",              d_tanh(0),         1.0)
    check("d_tanh(1)=0",              d_tanh(1),         0.0)
    check("d_tanh(0.5)=0.75",         d_tanh(0.5),       0.75)

    # ----------------------------------------------------------------
    # relu
    # ----------------------------------------------------------------
    check("relu(2)=2",                f_relu(2),         2.0)
    check("relu(0)=0",                f_relu(0),         0.0)
    check("relu(-3)=0",               f_relu(-3),        0.0)

    # ----------------------------------------------------------------
    # d_relu (lavora sul valore pre-attivazione)
    # ----------------------------------------------------------------
    check("d_relu(1)=1",              d_relu(1),         1.0)
    check("d_relu(0)=0",              d_relu(0),         0.0)
    check("d_relu(-1)=0",             d_relu(-1),        0.0)

    # ----------------------------------------------------------------
    # leaky_relu (alpha=0.1)
    # ----------------------------------------------------------------
    check("leaky_relu(2, 0.1)=2",     f_leaky_relu(2,  0.1),  2.0)
    check("leaky_relu(0, 0.1)=0",     f_leaky_relu(0,  0.1),  0.0)
    check("leaky_relu(-2, 0.1)=-0.2", f_leaky_relu(-2, 0.1), -0.2)
    check("leaky_relu(-2, 0.01)=-0.02", f_leaky_relu(-2, 0.01), -0.02)

    # ----------------------------------------------------------------
    # d_leaky_relu (lavora sul valore pre-attivazione)
    # ----------------------------------------------------------------
    check("d_leaky_relu(1,0.1)=1",    d_leaky_relu(1,  0.1),  1.0)
    check("d_leaky_relu(-1,0.1)=0.1", d_leaky_relu(-1, 0.1),  0.1)
    check("d_leaky_relu(0,0.2)=alpha", d_leaky_relu(0,  0.2),  0.2)

    # ----------------------------------------------------------------
    # apply_activation dispatcher
    # ----------------------------------------------------------------
    check("dispatch sigmoid(0)=0.5",      apply_activation(0, "sigmoid",    0),   0.5)
    check("dispatch tanh(0)=0",           apply_activation(0, "tanh",       0),   0.0)
    check("dispatch relu(-1)=0",          apply_activation(-1,"relu",       0),   0.0)
    check("dispatch leaky_relu(-1,0.1)",  apply_activation(-1,"leaky_relu", 0.1),-0.1)
    check("dispatch unknown=linear",      apply_activation(3, "unknown",    0),   3.0)
}
