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
	else {
		print "[WARNING]: Funzione di attivazione non trovata:", function_name, " - utilizzo della lineare" > "/dev/stderr"
		return x
	}
}

# Funzione per DERIVATIVE:
# Nota:
# - sigmoid, tanh 	: lavorano con l'output post-attivazione.
# - relu e leaky_relu 	: lavorano con il valore pre-attivazione.
function apply_activation_derivative(x, function_name, alpha) {
	if (function_name == "sigmoid")         return d_sigmoid(x)
    else if (function_name == "tanh")       return d_tanh(x)
    else if (function_name == "relu")       return d_relu(x)
    else if (function_name == "leaky_relu") return d_leaky_relu(x, alpha)
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
function compute_output_delta(output, target, activation, loss, alpha,    error, delta, d_activation) {
	error = output - target

	if (loss == "ce" && activation == "sigmoid") {
		logmesg(debug_backward, "[DEBUG] compute_output_delta: error=" error " delta=" error " loss=" loss "\n")
		return error
	}

	if (loss == "ce" && activation != "sigmoid") {
		logmesg(debug_backward, "[WARN] CE richiesta ma activation=" activation " non supporta CE, uso MSE\n")
	}
	d_activation = apply_activation_derivative(output, activation, alpha)
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
