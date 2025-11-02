# lib/activation.awk

# Funzione di attivazione astratta
function activation(x, name,    y) {
    if (name == "sigmoid") {
        y = sigmoid(x)
    } else if (name == "relu") {
        y = relu(x)
    } else if (name == "tanh") {
        y = tanh(x)
    } else {
        print "[ERROR] Funzione di attivazione sconosciuta:", name > "/dev/stderr"
        y = x  # fallback: identità
    }
    return y
}

# Derivata della funzione di attivazione
function activation_derivative(x, name,    y) {
    if (name == "sigmoid") {
        y = dsigmoid(x)
    } else if (name == "relu") {
        y = drelu(x)
    } else if (name == "tanh") {
        y = dtanh(x)
    } else {
        print "[ERROR] Derivata sconosciuta:", name > "/dev/stderr"
        y = 1
    }
    return y
}

# ----------------------
# Funzioni specifiche
# ----------------------

function sigmoid(x) {
    return 1 / (1 + exp(-x))
}

# NB: sigmoid(x) è già stato applicato prima
function dsigmoid(a,   x) {
    x = sigmoid(a)
    return x * (1 - x)
}

function relu(x) {
    return (x > 0 ? x : 0)
}

function drelu(a,   x) {
    x = relu(a)
    return (x > 0 ? 1 : 0)
}

function tanh(x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

function dtanh(a,   x) {
    x = tanh(a)
    return 1 - x * x  # assume x = tanh(x)
}

