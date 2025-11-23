### lib/framework/utils-math.awk
#
# Funzioni matematiche comuni, inizializzazione pesi e funzioni di errore
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

#########################################################################
# CLAMPING GRADIENT:
#########################################################################

# Restituisce il gradiente clampato:
function clip_gradient(debug_update, gradient, gradient_clip) {
	# Se max_value <= 0 --> Disabilitato quindi no clamping
	if (gradient_clip <= 0.0) {
		return gradient
	}

	# Clippa il MAX
	if (gradient > gradient_clip) {
		logmesg(debug_update, "[DEBUG] update: gradient = " + gradient " clipped to: " gradient_clip "\n")
		return gradient_clip
	}

	# Clippa il MIN:
	if (gradient < -gradient_clip) {
		logmesg(debug_update, "[DEBUG] update: gradient = " + gradient " clipped to: " -gradient_clip "\n")
		return -gradient_clip
	}

	# Altrimenti niente:
	logmesg(debug_update, "[DEBUG] update: gradient = " gradient ", clipping not needed\n")
	return gradient
}


#########################################################################
# INIT PESI:
#########################################################################

### 1. Dispatcher per inizializzazione pesi
function init_weight(method, fan_in, fan_out) {
    if (method == "xavier") return init_xavier(fan_in, fan_out)
    else if (method == "he") return init_he(fan_in, fan_out)
    else return init_random_uniform(fan_in, fan_out)
}

### 2. Metodi di inizializzazione
function init_random_uniform(fan_in, fan_out) {
    return rand() - 0.5   # range [-0.5, 0.5]
}

function init_xavier(fan_in, fan_out,    limit) {
    limit = sqrt(6 / (fan_in + fan_out))
    return rand() * 2 * limit - limit
}

function init_he(fan_in, fan_out,    limit) {
    limit = sqrt(2 / fan_in)
    return rand() * 2 * limit - limit
}

################################################################################
### 3. Funzioni matematiche di supporto
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

### 4. Funzioni di errore

## Dispatcher per la funzione di errore:
function calculate_error(error_function, target, output, num_outputs) {
	# La Cross-Entropy:
	if (error_function == "cross_entropy") return error_cross_entropy(target, output, num_outputs)
	# MSE:
	else return error_mse(target, output, num_outputs)
}

# Mean Squared Error
function error_mse(target, output, n,    i, sum) {
    sum = 0
    for (i = 1; i <= n; i++) sum += (target[i] - output[i]) ^ 2
    return sum / n
}

# Cross-Entropy Error (per sigmoid)
function error_cross_entropy(target, output, n,    i, sum, eps) {
    sum = 0; eps = 1e-15
    for (i = 1; i <= n; i++) {
        sum += -(target[i] * log(output[i] + eps) + (1 - target[i]) * log(1 - output[i] + eps))
    }
    return sum / n
}

