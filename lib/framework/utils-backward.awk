#
# Definiamo la funzione che serve per calcolare i :
# - delta
# - gradient
# per tutti i layers.
#
# Parametri:
#   num_samples      = numero di campioni
#   num_layers       = numero di layer
#   dataset_meta     = metadati dataset (per capire num_outputs, ecc.)
#   dataset_targets  = target (sample, output_idx)
#   layer_meta       = metadati dei layer
#   layer_weights    = pesi dei layer (layer, neuron, input)
#   layer_outputs    = output dei layer (layer, sample, neuron)
#   layer_deltas     = (by ref) errori/delta calcolati per ogni layer, sample, neuron
#

# Esegue un singolo step di backward, una volta fatto il forward:
function backward_pass(dataset_meta, dataset_targets, layer_meta, layer_weights, layer_output, layer_preactivation, layer_deltas, 
			num_samples, num_layers, num_neurons, sample, layer_id, neuron, output, target, error, 
			d_activation, activation_function, delta, num_neurons_next, layer_id_next, sum_error, 
			neuron_next, weight_next, delta_next, ds_info, layer_info, next_layer_info) {
	# Estraiamo i dati di partenza:
	get_dataset_info(dataset_meta, ds_info)
	num_samples = ds_info["num_samples"]
	num_layers = get_num_layers(layer_meta)
	
	# Tipo funzione di LOSS: [mse | ce] ce = Cross Entropy, mse = default
	if (loss_function == "") {
		loss_function = "mse"
	}
	
	# STEP 1 - OUTPUT LAYER (calcolo differente da HIDDEN)
	# Partenza da output layer:
	layer_id = num_layers
	get_layer_info(layer_meta, layer_id, layer_info)
	num_neurons = layer_info["num_neurons"]
	activation_function = layer_info["activation"]
	for (sample = 1; sample<=num_samples; sample++) {
		
		# Numero neuroni del layer attuale:
		for (neuron=1; neuron<=num_neurons; neuron++) {
			# Prendiamo il valore di output del neurone:
			output = layer_output[layer_id, sample, neuron]
			
			# Prendiamo il TARGET:
			target = dataset_targets[sample, neuron]

			# Salvataggio dato in layer_deltas[layer, sample, neuron]
			delta = compute_output_delta(output, target, activation_function, loss_function)
			layer_deltas[layer_id, sample, neuron] = delta
			
			# Debug dettagliato:
			logmesg(debug_backward, "[DEBUG] backward: OUT_LAYER sample=" sample \
			" neuron=" neuron " target=" target " output=" output \
			" delta=" delta " loss=" loss_function "\n")
		}
		# Aggiornamento della matrice deltas:
		layer_deltas[layer_id, sample, 0] = num_neurons	
	}
	# Salvo le righe del layer:
	layer_deltas[layer_id, 0, 0] = num_samples

	
	# STEP 2 - HIDDEN LAYERS (calcolo differente da OUTPUT)	
	for (layer_id = num_layers-1; layer_id >= 1; layer_id--) {
		layer_id_next = layer_id + 1

		# Estraggo metadati del layer corrente:
		get_layer_info(layer_meta, layer_id, layer_info)
		num_neurons = layer_info["num_neurons"]
		activation_function = layer_info["activation"]

		# Metatadi del NEXT_LAYER:
		get_layer_info(layer_meta, layer_id_next, next_layer_info)
		num_neurons_next = next_layer_info["num_neurons"]

		# Calcolo delta per ogni sample:		
		for (sample=1; sample<=num_samples; sample++) {
			
			# Numero neuroni layer attuale:
			for (neuron=1; neuron<=num_neurons; neuron++) {
				# Calcolo dei delta HIDDEN:
				sum_error = 0
			
				# Cicliamo sui neuroni del NEXT_LAYER:
				for (neuron_next=1; neuron_next<=num_neurons_next; neuron_next++) {
					weight_next = layer_weights[layer_id_next, neuron_next, neuron]
					delta_next = layer_deltas[layer_id_next, sample, neuron_next]
					
					# Somma pesata:						
					sum_error += delta_next * weight_next
				}
				# Fine neuron_next	

				# Prendiamo l'output del neurone e calcoliamo la derivata:
				# Caso GENERICO : output = layer_output[layer_id, sample, neuron]
				# Scegliamo il valore corretto a seconda della funzione di ATTIVAZIONE:
				output = 0.0
				if (activation_function == "relu" || activation_function == "leaky_relu") {
					# ReLU family: usa PRE-ACTIVATION
					output = layer_preactivation[layer_id, sample, neuron]
				}
				else {
					# Sigmoid/Tanh: usa POST-ACTIVATION
					output = layer_output[layer_id, sample, neuron]
				}
				# Applico sempre la derivata:
				d_activation = apply_activation_derivative(output, activation_function)
			
				# Calcolo del delta per neurone HIDDEN:
				delta = sum_error * d_activation

				# Salvataggio nella matrice:
				layer_deltas[layer_id, sample, neuron] = delta	

				# Debug dettagliato:
				logmesg(debug_backward, "[DEBUG] backward: HIDDEN layer=" layer_id \
				" sample=" sample " neuron=" neuron " sum_error=" sum_error \
				" d_activation=" d_activation " delta=" delta "\n")
			}
			# Fine neurons
			layer_deltas[layer_id, sample, 0] = num_neurons
		}
		# Aggiorno anche queste righe:
		layer_deltas[layer_id, 0, 0] = num_samples
	}
	# Salvo le righe della delta:
	layer_deltas[0, 0, 0] = num_layers
	# FINE:
}
