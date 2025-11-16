###############################################################################
# utils-loss.awk
#
# Funzioni per calcolare la loss (MSE, Cross-Entropy, ecc.)
#
# Convenzioni:
# - loss    : "mse" oppure "ce" (binary cross-entropy)
# - activation: funzione di attivazione dell'OUTPUT ("sigmoid", "tanh", ...)
#
# Nota:
# - CE è supportata SOLO per activation="sigmoid".
#   Per tutte le altre activation facciamo fallback a MSE.
###############################################################################

# Loss per un singolo neurone di output
# Parametri:
#   output     = y  (valore attivato del neurone)
#   target     = t  (valore target, tipicamente 0 o 1)
#   activation = funzione di attivazione dell'OUTPUT ("sigmoid", "tanh", ...)
#   loss       = "mse" oppure "ce"
#
function compute_sample_loss(output, target, activation, loss,
                             eps, y, t, one_minus_y, diff) {
    # Binary Cross-Entropy per sigmoid:
    if (loss == "ce" && activation == "sigmoid") {
        # Protezione da log(0)
        eps = 1e-15
        if (output < eps)          y = eps
        else if (output > 1.0-eps) y = 1.0 - eps
        else                       y = output

        t = target
        one_minus_y = 1.0 - y

        # CE = - [ t*log(y) + (1-t)*log(1-y) ]
        return -(t * log(y) + (1.0 - t) * log(one_minus_y))
    }

    # Fallback: MSE = 1/2 * (output - target)^2
    diff = output - target
    return (0.5 * diff * diff)
}

###############################################################################
# compute_dataset_loss()
# Calcola la loss media sull'OUTPUT layer dell'intero dataset
#
# Parametri:
#   dataset_meta     : metadati dataset (num_samples, ...)
#   dataset_targets  : target[sample, neuron]
#   layer_meta       : metadati layer (num_layers, num_neurons, activation, ...)
#   layer_output     : output dei layer[layer, sample, neuron]
#   loss_function    : "mse" oppure "ce"
#
# Restituisce:
#   loss media per neurone di output (valore scalare)
###############################################################################
function compute_dataset_loss(dataset_meta, dataset_targets,
                              layer_meta, layer_output,
                              loss_function,
                              num_samples, num_outputs, num_layers,
                              activation_output,
                              sample, neuron,
                              output, target, sum_loss) {

    # Recuperiamo le info di base:
    num_samples       = dataset_meta["num_samples"]
    num_layers        = layer_meta[0, 0, 0]
    num_outputs       = layer_meta[num_layers, "num_neurons"]
    activation_output = layer_meta[num_layers, "activation"]

    sum_loss = 0.0

    for (sample = 1; sample <= num_samples; sample++) {
        for (neuron = 1; neuron <= num_outputs; neuron++) {

            output = layer_output[num_layers, sample, neuron]
            target = dataset_targets[sample, neuron]

            sum_loss += compute_sample_loss(output, target,
                                            activation_output,
                                            loss_function)
        }
    }

    # Loss media per neurone di output
    return sum_loss / (num_samples * num_outputs)
}

###############################################################################
# compute_mse()
# Calcola la Mean Squared Error sui campioni e neuroni di output
#
# Parametri:
#   num_samples       = numero campioni
#   num_outputs       = numero neuroni nell'output layer
#   num_layers        = numero layer totali
#   dataset_targets   = array target[sample, neuron]
#   layer_output      = output dei layer[layer, sample, neuron]
###############################################################################
function compute_mse(dataset_meta, dataset_targets, layer_meta, layer_output,
			num_samples, num_outputs, num_layers, sample, neuron, output, target, mse, sum_sq_error) {

	# Preleviamo i dati che ci servono:
	num_samples = dataset_meta["num_samples"]
	num_outputs = dataset_meta["num_outputs"]
	
	num_layers = layer_meta[0, 0, 0]

	# Debug delle metriche:
	logmesg(debug_metrics, "[DEBUG] metrics: num_samples = "num_samples"\n")
	logmesg(debug_metrics, "[DEBUG] metrics: num_outputs = "num_outputs"\n")
	logmesg(debug_metrics, "[DEBUG] metrics: num_layers = "num_layers"\n")

	# Inizializziamo la sommatoria:	
    	sum_sq_error = 0

	# Per ogni sample calocliamo:
	for (sample = 1; sample <= num_samples; sample++) {

        	for (neuron = 1; neuron <= num_outputs; neuron++) {

			# Output del layer finale:
            		output = layer_output[num_layers, sample, neuron]
			# Target del dataset:
            		target = dataset_targets[sample, neuron]

			# Sommiamo per la somma totale:
            		sum_sq_error += (target - output) ^ 2
		}
    	}

	# Restituiamo un valore e non una matrice:
	mse = sum_sq_error / (num_samples * num_outputs)
    	return mse
}