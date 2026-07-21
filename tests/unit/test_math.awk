#!/usr/bin/gawk -f
#
# Test unitari AWK per utils-math.awk.
#

function nearly_eq(a, b, tol,    diff) {
    tol  = (tol == "" ? 1e-9 : tol)
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
    # min / max
    # ----------------------------------------------------------------
    check("min(3,5)=3",    min(3,5),   3)
    check("min(-1,0)=-1",  min(-1,0), -1)
    check("min(4,4)=4",    min(4,4),   4)

    check("max(3,5)=5",    max(3,5),   5)
    check("max(-1,0)=0",   max(-1,0),  0)
    check("max(4,4)=4",    max(4,4),   4)

    # ----------------------------------------------------------------
    # clamp
    # ----------------------------------------------------------------
    check("clamp(5,0,10)=5",   clamp(5,  0, 10), 5)
    check("clamp(-1,0,10)=0",  clamp(-1, 0, 10), 0)
    check("clamp(15,0,10)=10", clamp(15, 0, 10), 10)
    check("clamp(0,0,0)=0",    clamp(0,  0,  0), 0)

    # ----------------------------------------------------------------
    # normalize_vector
    # ----------------------------------------------------------------
    v[1] = 3; v[2] = 4; v[0] = 2
    normalize_vector(v, 2)
    check("normalize [3,4]: v[1]=0.6", v[1], 0.6, 1e-9)
    check("normalize [3,4]: v[2]=0.8", v[2], 0.8, 1e-9)

    # vettore zero: normalize non divide (norma=0, nessun crash)
    z[1] = 0; z[2] = 0; z[0] = 2
    normalize_vector(z, 2)
    check("normalize [0,0]: no crash v[1]=0", z[1], 0.0)
    check("normalize [0,0]: no crash v[2]=0", z[2], 0.0)

    # ----------------------------------------------------------------
    # random_gaussian: verifica media e stddev su N campioni
    # ----------------------------------------------------------------
    srand(42)
    N = 5000; mean_target = 2.0; std_target = 0.5
    sum = 0; sum2 = 0
    for (i = 1; i <= N; i++) {
        x = random_gaussian(mean_target, std_target)
        sum  += x
        sum2 += x * x
    }
    mean_got = sum / N
    std_got  = sqrt(sum2/N - mean_got^2)
    check("gaussian mean~2.0",   mean_got, mean_target, 0.05)
    check("gaussian stddev~0.5", std_got,  std_target,  0.05)
}
