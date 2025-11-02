#
# Proviamo a fare un forward step di una rete neurale, che abbia in questo file
# la possibilita' di lanciare il suo forward step:
#

# Funzione che esegue il forward step:
function forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output,    
				layer_id, num_samples, sample, input, num_inputs, input_array, bias_index, 
				activation_function, z, neuron, num_neurons, pred, num_outputs) {
	# Procediamo estraendo alcuni dati:	
	num_samples = dataset_meta["num_samples"]
	num_inputs = dataset_meta["num_inputs"]
	num_outputs = dataset_meta["num_outputs"]

	# Stampiamo tutti i parametri prelevati prima:
	logmesg(debug_forward, "[DEBUG] forward: num_samples = "num_samples"\n")
	logmesg(debug_forward, "[DEBUG] forward: num_inputs = "num_inputs"\n")
	logmesg(debug_forward, "[DEBUG] forward: num_outputs = "num_outputs"\n")

	# Devo ciclare su tutti i sample del dataset input (comprensivo di bias):
	for (sample=1; sample<=num_samples; sample++) {
		logmesg(debug_forward, "[DEBUG] forward: starting cycle for sample: "sample"/"num_samples"\n")
		
		# Copia del sample su un array:
		delete input_array
		copy_matrix_row_to_array(dataset_weights, sample, input_array)
		logmesg(debug_forward, "[DEBUG] forward: initialized input_array: "array_to_string(input_array)"\n")

		# Ora dobbiamo ciclare su tutti i layer per procedere al forward pass:
		logmesg(debug_forward, "[DEBUG] forward: starting cycle on num_layers = "num_layers"\n")
		for (layer_id=1; layer_id<=num_layers; layer_id++) {
			# Forward pass on layer:
			logmesg(debug_forward, "[DEBUG] forward: starting forward pass on layer = "layer_id"\n")

			# Estraiamo la funzione di attivazione del layer:
			activation_function = layer_meta[layer_id, "activation"]
			num_neurons = layer_meta[layer_id, "num_neurons"]
			num_inputs = layer_meta[layer_id, "num_inputs"]
			
			# Stampiamo dati sul layer attuale:
			logmesg(debug_forward, "[DEBUG] forward: layer"layer_id" activation_function = "activation_function"\n")
			logmesg(debug_forward, "[DEBUG] forward: layer"layer_id" num_neurons = "num_neurons"\n")
			logmesg(debug_forward, "[DEBUG] forward: layer"layer_id" num_inputs = "num_inputs"\n")

			# STEP 1 - Calcolo outputs del layer con funzione di attivazione.
			for (neuron=1; neuron<=num_neurons; neuron++) {
				# Calcoliamo attivazione come sommatoria di input * weights:
				z = 0
				# Qua stiamo includendo il bias:
				for (input=1; input<=num_inputs; input++) {
					z += (input_array[input] * layer_weights[layer_id, neuron, input])
				}
				
				# Ora attiviamo:
				layer_output[layer_id, sample, neuron] = apply_activation(z, activation_function)
				logmesg(debug_forward, "[DEBUG] forward: layer_output["layer_id", "sample", "neuron"] = "layer_output[layer_id, sample, neuron]"\n")
			}
			# Settiamo dimensione layer_output:
			layer_output[layer_id, sample, 0] = num_neurons 

			# STEP 2 - Preparazione degli inputs per layer NEXT 
                        delete input_array
                        for (neuron=1; neuron<=num_neurons; neuron++) {
                        	# Copio gli output di questo layer nel successivo:
                                input_array[neuron] = layer_output[layer_id, sample, neuron]
                        }

			# STEP 3 - Inseriamo il bias se il layer lo richiede. [AL MOMENTO SEMPRE]
                        # Verifichiamo se dobbiamo inserire il bias (tipicamente sì)
                        if (layer_meta[layer_id, "has_bias"]) {
                        	bias_index = num_neurons +1
                                input_array[bias_index] = 1.0
                                # Aggiornamento dimensione:
                                input_array[0] = bias_index
                        }
                        # Verifichiamo se stampiamo gli input del nuovo layer:
                        logmesg(debug_forward, "[DEBUG] forward: inputs for NEXT_LAYER = "array_to_string(input_array)"\n")

			# Fine forward pass:
			logmesg(debug_forward, "[DEBUG] forward: ending forward pass on layer = "layer_id"\n")
		}
		# Aggiusto il numero di righe degli output:
		layer_output[0, 0, 0] = num_layers

		# Fine del sample:
		logmesg(debug_forward, "[DEBUG] forward: ending cycle for sample: "sample"/"num_samples"\n")
	}
	# Fine forward pass.

	# OUTPUT FORWARD PASS:
	
	# Stampa OUTPUT:
	if (print_result) {
		# Stampiamo solo se richiesto:
		for (sample = 1; sample<=num_samples; sample++) {
			printf("[RESULT] Sample %d -> pred = ", sample)

			# Output dell'ultimo layer (con più neuroni in caso ce ne fossero)
			num_neurons = layer_meta[num_layers, "num_neurons"]
			for (neuron = 1; neuron<=num_neurons; neuron++) {
				pred = layer_output[num_layers, sample, neuron]
				printf("%.6f ", pred)
			}

			# Target atteso:
			printf("| target = ")
			for (neuron = 1; neuron<=num_outputs; neuron++) {
				printf("%s ", dataset_targets[sample, neuron])
			}

			# Fine riga:
			printf("\n")
		}
	}
}
