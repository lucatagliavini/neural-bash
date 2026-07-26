#
# Il file serve per fornire le funzioni di attivazione per la rete neurale.
# Sulla base di quello che viene passato come nome della funzione:
# - sigmoid
# - relu
# - tanh
# - leaky_relu
# 
# il fallback è : linear
#
# Cross-Entropy:
# --------------
# CE = −[t * log(y) + (1−t) * log(1−y)]
# 
# t: target
# y: output interpretato come probabilità quindi 0.0 <= y <= 1.0
#

# Funzione per ATTIVAZIONE:
# Parametri:
# - x = valore da attivare
# - function_name = Nome funzione di attivazione
function apply_activation(x, function_name, alpha) {
	if (function_name == "sigmoid")		return f_sigmoid(x)
	else if (function_name == "tanh")	return f_tanh(x)
	else if (function_name == "relu")	return f_relu(x)
	else if (function_name == "leaky_relu")	return f_leaky_relu(x, alpha)
	else if (function_name == "softmax")	return x  # softmax è vettoriale, applicata dopo
	else if (function_name == "linear")	return x
	else {
		print "[WARNING]: Funzione di attivazione non trovata:", function_name, " - utilizzo della lineare" > "/dev/stderr"
		return x
	}
}

# Applica softmax in-place su layer_output[layer_id, sample, 1..num_neurons].
# Sottrae il massimo per stabilità numerica (log-sum-exp trick).
function apply_softmax(layer_output, layer_id, sample, num_neurons,    i, max_z, sum_exp, z) {
	max_z = layer_output[layer_id, sample, 1]
	for (i = 2; i <= num_neurons; i++) {
		z = layer_output[layer_id, sample, i]
		if (z > max_z) max_z = z
	}
	sum_exp = 0
	for (i = 1; i <= num_neurons; i++) {
		layer_output[layer_id, sample, i] = exp(layer_output[layer_id, sample, i] - max_z)
		sum_exp += layer_output[layer_id, sample, i]
	}
	for (i = 1; i <= num_neurons; i++) {
		layer_output[layer_id, sample, i] /= sum_exp
	}
}

# Funzione per DERIVATIVE:
# Riceve sia preact (z, pre-attivazione) che postact (a, post-attivazione).
# Ogni derivata usa il valore matematicamente corretto:
# - sigmoid, tanh    : usano postact (derivata espressa in termini di f(z), più efficiente)
# - relu, leaky_relu : usano preact  (segno di z determina la derivata)
function apply_activation_derivative(preact, postact, function_name, alpha) {
	if (function_name == "sigmoid")         return d_sigmoid(postact)
    else if (function_name == "tanh")       return d_tanh(postact)
    else if (function_name == "relu")       return d_relu(preact)
    else if (function_name == "leaky_relu") return d_leaky_relu(preact, alpha)
    else if (function_name == "softmax")    return 1.0  # delta già calcolato come (p-y) in compute_output_delta
    else if (function_name == "linear")     return 1.0
    else {
        print "[WARNING]: Funzione di deattivazione non trovata:" , function_name, " - utilizzo della lineare" > "/dev/stderr"
        return 1.0
    }
}

# Funzione per calcolo OUTPUT_LAYER con Cross-Entropy:
# funziona bene solo per sigmoid, perché la funzione deve
# avere un dominio [0, 1].
# Parametri:
# - output: 	valore attivato del neurone (y)
# - target:		valore target (t)
# - activation:	funzione di attivazione usata su output ("sigmoid", "relu", ...)
# - loss:		funzione di loss ("mse", "ce", ...)
#
# Convenzione: error = output - target (coerente con update: w -= lr * gradient)
#
function compute_output_delta(preact, output, target, activation, loss, alpha,    error, delta, d_activation) {
	error = output - target

	# Binary CE con sigmoid: delta = output - target (semplificazione analitica)
	if (loss == "ce" && activation == "sigmoid") {
		logmesg(debug_backward, "[DEBUG] compute_output_delta: error=" error " delta=" error " loss=" loss "\n")
		return error
	}

	# Categorical CE con softmax: delta = output - target (stessa forma, derivata combinata CCE+softmax)
	if (loss == "cce" && activation == "softmax") {
		logmesg(debug_backward, "[DEBUG] compute_output_delta: error=" error " delta=" error " loss=cce\n")
		return error
	}

	if (loss == "ce" && activation != "sigmoid") {
		logmesg(debug_backward, "[WARN] CE richiesta ma activation=" activation " non supporta CE, uso MSE\n")
	}
	if (loss == "cce" && activation != "softmax") {
		logmesg(debug_backward, "[WARN] CCE richiesta ma activation=" activation " non è softmax, uso MSE\n")
	}
	d_activation = apply_activation_derivative(preact, output, activation, alpha)
	delta = error * d_activation
	logmesg(debug_backward, "[DEBUG] compute_output_delta: error=" error " delta=" delta " loss=" loss "\n")
	return delta
}

# ========================================================================
# Funzioni di ATTIVAZIONE:
# ========================================================================

# Sigmoid: output tra 0 e 1
function f_sigmoid(x) {
	return 1.0 / (1.0 + exp(-x))
}

# Tanh: output tra -1 e 1
function f_tanh(x,    ex, enx) {
	ex = exp(x); enx = exp(-x)
	return (ex - enx) / (ex + enx)
}

# RelU: zero per input negativi
function f_relu(x) {
	return (x > 0 ? x : 0)
}

# Leaky RelU: piccola pendenza per input negativi
function f_leaky_relu(x, alpha) {
	return (x > 0 ? x : alpha * x)
}


# ========================================================================
# Funzioni di DERIVATIVE:
# ========================================================================

# D-Sigmoid: Input "y" = output già attivato (sigmoid(x))
function d_sigmoid(y) {
	return y * (1.0 - y)
}

# D-Tanh: Input "y" = output già attivato (tanh(x))
function d_tanh(y) {
	return 1.0 - (y * y)
}

# D-RelU: Input "x" valore pre attivazione!
function d_relu(x) {
	return (x > 0 ? 1.0 : 0.0)
}

# Leaky RelU: Input "x" valore pre-attivazione!
function d_leaky_relu(x, alpha) {
	return (x > 0 ? 1.0 : alpha)
}
