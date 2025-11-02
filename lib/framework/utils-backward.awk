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
function backward_pass(dataset_meta, dataset_targets, layer_meta, layer_weights, layer_output, layer_deltas,    
			num_samples, num_layers, num_neurons, sample, layer_id, neuron, output, target, error, 
			d_activation, activation_function, delta, num_neurons_next, layer_id_next, sum_error, 
			neuron_next, weight_next, delta_next) {
	# Estraiamo i dati di partenza:
	num_samples = dataset_meta["num_samples"]
	num_layers = layer_meta[0, 0, 0]

	
	# STEP 1 - OUTPUT LAYER (calcolo differente da HIDDEN)
	# Partenza da output layer:
	layer_id = num_layers
	num_neurons = layer_meta[layer_id, "num_neurons"]
	activation_function = layer_meta[layer_id, "activation"]
	for (sample = 1; sample<=num_samples; sample++) {
		
		# Numero neuroni del layer attuale:
		for (neuron=1; neuron<=num_neurons; neuron++) {
			# Prendiamo il valore di output del neurone:
			output = layer_output[layer_id, sample, neuron]
			
			# Prendiamo il TARGET:
			target = dataset_targets[sample, neuron]

			# Errore: (target - output)
			error = target - output

			# Calcoliamo la derivata della funzione di attivazione [DA OUTPUT]
			d_activation = apply_activation_derivative(output, activation_function)

			# Salvataggio dato in layer_deltas[layer, sample, neuron]
			delta = error * d_activation
			layer_deltas[layer_id, sample, neuron] = delta
			
			# Debug dettagliato:
			logmesg(debug_backward, "[DEBUG] backward: OUT_LAYER sample=" sample \
			" neuron=" neuron " target=" target " output=" output " error=" error \
			" d_activation=" d_activation " delta=" delta "\n")
		}
		# Aggiornamento della matrice deltas:
		layer_deltas[layer_id, sample, 0] = num_neurons	
	}
	# Salvo le righe del layer:
	layer_deltas[layer_id, 0, 0] = num_samples

	
	# STEP 2 - HIDDEN LAYERS (calcolo differente da OUTPUT)	
	for (layer_id = num_layers-1; layer_id >= 1; layer_id--) {
		layer_id_next = layer_id + 1

		# Estraggo il numero neuroni del layer:
		num_neurons = layer_meta[layer_id, "num_neurons"]
		activation_function = layer_meta[layer_id, "activation"]
		# Numero neuroni del NEXT_LAYER:
		num_neurons_next = layer_meta[layer_id_next, "num_neurons"]

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
				output = layer_output[layer_id, sample, neuron]
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
