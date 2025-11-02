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
