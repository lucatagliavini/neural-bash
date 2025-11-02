# nnet-backward.awk
# Calcola i delta e i gradienti del layer di output
# Richiede: -f activation.awk -f math.awk
# Parametri richiesti:
#   -v output_file=...           output del layer corrente (es. layer1.out)
#   -v target_file=...           dataset completo con target
#   -v activation_function=...   nome funzione attivazione (es. sigmoid)
#   -v delta_output_file=...     dove scrivere i delta (es. layer1-delta.txt)
# Opzionali:
#   -v input_file=...            se definito, calcola anche gradienti
#   -v gradient_output_file=...  dove scrivere gradienti
#   -v learning_rate=...         default: 0.1
#   -v debug=1                   per log su stderr

BEGIN {
    debug = (debug ? 1 : 0)
    output_idx = 0

    # Carica pesi e delta del layer successivo (solo se hidden layer)
    has_next = 0
    if (next_weights_file != "" && next_delta_file != "") {
        has_next = load_next_weights(next_weights_file, next_weights, next_activation, next_rows, next_cols)
        has_next += load_next_deltas(next_delta_file, next_deltas)
        if (debug && has_next >= 2) print "# Loaded next layer weights and deltas" > "/dev/stderr"
    }
}

# Funzione: carica pesi del layer successivo in next_weights[row, col]
function load_next_weights(file, weights,   line, r, fields, j) {
    r = 0
    while ((getline line < file) > 0) {
        if (line ~ /^ACTIVATION=/ || line ~ /^ROWS=/ || line ~ /^COLS=/) continue
        ++r
        split(line, fields, /[ \t]+/)
        for (j = 1; j <= length(fields); j++) {
            weights[r, j] = fields[j]
        }
    }
    close(file)
    return 1
}

# Funzione: carica i delta del layer successivo
function load_next_deltas(file, deltas,   line, r, fields, j) {
    r = 0
    while ((getline line < file) > 0) {
        ++r
        split(line, fields, /[ \t]+/)
        for (j = 1; j <= length(fields); j++) {
            deltas[r, j] = fields[j]
        }
    }
    close(file)
    return 1
}

{
    ++output_idx
    y = $1

    # Input della stessa riga
    getline input_line < input_file
    split(input_line, input, /[ \t]+/)

    # Caso output layer: usare target diretto
    if (target_file != "") {
        getline target_line < target_file
        t = target_line + 0
        err = y - t
        dact = activation_derivative(y, activation_function)
        delta = err * dact
        if (debug) printf "# DELTA r=%d y=%.6f t=%.6f err=%.6f d_act=%.6f delta=%.6f\n", output_idx, y, t, err, dact, delta > "/dev/stderr"
    }
    # Caso hidden layer: propagazione del gradiente dai delta successivi
    else {
        sum = 0
        for (k = 1; next_deltas[k,1] != ""; k++) {
            # Neurone k del next layer → considera tutti i neuroni collegati
            # qui stiamo nel neurone j = output_idx
            weight_kj = next_weights[k, output_idx]
            delta_k = next_deltas[k, 1]
            sum += delta_k * weight_kj
        }
        dact = activation_derivative(y, activation_function)
        delta = dact * sum
        if (debug) printf "# HIDDEN DELTA r=%d y=%.6f d_act=%.6f sum=%.6f delta=%.6f\n", output_idx, y, dact, sum, delta > "/dev/stderr"
    }

    print delta >> delta_output_file

    # Calcola gradienti: delta * input[i]
    line = ""
    for (i = 1; i <= length(input); i++) {
        grad = delta * input[i]
        line = line grad " "
        if (debug) printf "# GRAD r=%d i=%d delta=%.6f input=%.6f grad=%.6f\n", output_idx, i, delta, input[i], grad > "/dev/stderr"
    }
    sub(/[ \t]+$/, "", line)
    print line >> gradient_output_file
}

