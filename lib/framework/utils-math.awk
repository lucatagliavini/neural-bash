### lib/framework/utils-math.awk
#
# Funzioni matematiche comuni.
#

# Gestione minimo massimo e CLAMPING:
function min(val1, val2) {
	return (val1 < val2 ? val1 : val2)	
}

# Massimo:
function max(val1, val2) {
	return (val1 > val2 ? val1 : val2)
}

# Clamping:
function clamp(value, min_value, max_value) {
	return max(min_value, min(value, max_value))
}


################################################################################
# Funzioni matematiche di supporto
################################################################################

# Normalizza un vettore in-place (usato per feature o pesi)
function normalize_vector(v, n,    i, norm) {
    norm = 0
    for (i = 1; i <= n; i++) norm += v[i] ^ 2
    norm = sqrt(norm)
    if (norm > 0) for (i = 1; i <= n; i++) v[i] /= norm
}

# Random gaussiano (Box-Muller) per eventuali altre inizializzazioni
function random_gaussian(mean, stddev,    u1, u2, r, theta) {
    u1 = rand(); u2 = rand()
    r = sqrt(-2 * log(u1)) * stddev
    theta = 2 * 3.141592653589793 * u2
    return mean + r * cos(theta)
}


