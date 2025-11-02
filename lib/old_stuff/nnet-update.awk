# awk: lib/functions/nnet-update.awk
#
# Author: Luca Tagliavini
#
# Parametri:
# awk \
#  -v input_file="tmp/xor-test/input_only.txt" \   	# Per gli altri layer: -v input_file="tmp/xor-test/layer1_with_bias.tmp" \
#  -v grad_file="tmp/xor-test/layer1-grad.txt" \
#  -v weights_file="models/xor/layer1.txt" \
#  -v output_file="tmp/xor-test/layer1-updated.txt" \
#  -v learning_rate=0.1 \
#  -v debug=1 \
#  -f lib/functions/math.awk \
#  -f lib/functions/shared-functions.awk \
#  -f lib/functions/nnet-update.awk \
#  dummy
#
BEGIN {
    if (debug) print "# Starting update" > "/dev/stderr"

    ### === Carica pesi ===
    if (!read_weights(weights_file, weights, activation, nrows, ncols)) {
        print "Errore lettura pesi da " weights_file > "/dev/stderr"
        exit 1
    }
    activation = g_activation
    nrows = g_nrows
    ncols = g_ncols
    if (debug) printf "# Weights loaded: %dx%d (%s)\n", nrows, ncols, activation > "/dev/stderr"

    ### === Carica input ===
    num_inputs = read_matrix(input_file, input)
    if (debug) printf "# Input examples: %d\n", num_inputs > "/dev/stderr"

    ### === Carica gradienti ===
    num_grads = read_matrix(grad_file, grad)
    if (debug) printf "# Gradient rows: %d\n", num_grads > "/dev/stderr"

    ### === Aggiorna pesi (ma non li salviamo ancora) ===
    for (j = 1; j <= nrows; j++) {
        for (i = 1; i <= ncols; i++) {
            sum_grad = 0
            for (r = 1; r <= num_grads; r++) {
                sum_grad += grad[r, i]
            }
            mean_grad = sum_grad / num_grads
            old = weights[j, i]
            weights[j, i] = old - learning_rate * mean_grad
            if (debug) printf "# neuron=%d i=%d old=%f grad=%.6f new=%.6f\n", j, i, old, mean_grad, weights[j, i] > "/dev/stderr"
        }
    }
}

# Blocco vuoto: ignoriamo righe del file principale
{
    # Nessuna azione richiesta sul dummy
}

END {
    write_weights(output_file, weights, activation, nrows, ncols)
    if (debug) print "# Weights written to " output_file > "/dev/stderr"
}

