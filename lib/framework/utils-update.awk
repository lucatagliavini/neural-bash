#
# File che implementa l'ultimo "step" del training della Neural Netowrk: UPDATE
#

# Funzione di update dei pesi dalla matrice delta:
function update_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, layer_output, layer_deltas, learning_rate,
		layer_id, layer_id_prev, num_samples, num_layers, num_inputs, neuron, sample, delta, gradient, 
		input_id, input_value, weight_value) {
	
	# Recupero i dati necessari e loggo:
	num_samples = dataset_meta["num_samples"]
	num_layers = layer_meta[0, 0, 0]

	# Settiamo la learning rate:
	if (learning_rate == "" || learning_rate == 0) {
		learning_rate = 0.1
	}
	# Debug:
	logmesg(debug_update, "[DEBUG] update: num_samples = "num_samples"\n")	
	logmesg(debug_update, "[DEBUG] update: num_layers = "num_layers"\n")	
	logmesg(debug_update, "[DEBUG] update: learning_rate = "learning_rate"\n")

	# Cicliamo sui layer:
	logmesg(debug_update, "[DEBUG] update: starting cycle on "num_layers" layers\n")
	for (layer_id = 1; layer_id<=num_layers; layer_id++) {
		logmesg(debug_update, "[DEBUG] update: updating layer_id = "layer_id"\n")
	
		# Recuperiamo i dati del layer:
		num_neurons = layer_meta[layer_id, "num_neurons"]
		num_inputs = layer_meta[layer_id, "num_inputs"]

		# Debug:
		logmesg(debug_update, "[DEBUG] update: num_neurons = "num_neurons"\n")
		logmesg(debug_update, "[DEBUG] update: num_inputs = "num_inputs"\n")

		# Aggiornamento neuroni:
		for (neuron = 1; neuron<=num_neurons; neuron++) {

			# Aggiornamento per ogni input:
			for (input_id = 1; input_id<=num_inputs; input_id++) {
				
				# Sommiamo il gradiente su tutti i campioni
				gradient = 0
				for (sample = 1; sample<=num_samples; sample++) {
					
					# Delta calcolo:
					delta = layer_deltas[layer_id, sample, neuron]

					# Troviamo input corretto:
					if (layer_id == 1) {
						# Primo layer: input viene dal dataset (dataset_weights)
						input_value = dataset_weights[sample, input_id]	
					}
					else {
						# Layer successivi: input = output del layer precedente:
						layer_id_prev = layer_id - 1
						input_value = layer_output[layer_id_prev, sample, input_id]
					}

					# Aggiungo al gradiente:
					gradient += delta * input_value
				}
				# Fine ciclo SAMPLE.

				# Calcoliamo la media:
				gradient = gradient / num_samples

				# Aggiornamento del peso:
				weight_value = layer_weights[layer_id, neuron, input_id]
				layer_weights[layer_id, neuron, input_id] = weight_value + (learning_rate * gradient)

				# Debug apprendimento:
				logmesg(debug_update, "[DEBUG] update: layer=" layer_id " neuron=" neuron \
				" input=" input_id " weight_old=" weight_value " gradient=" gradient \
				" weight_new=" layer_weights[layer_id, neuron, input_id]"\n")
			}  
			# End ciclo INPUTS.
		}
		# End ciclo NEURONI del layer.
	}
	# End ciclo LAYERS.
}
