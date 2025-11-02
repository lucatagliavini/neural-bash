#
# Questo script awk, si propone di eseguire uno step di backpropagation in una struttura
# a layer che abbiamo implementato.
#
# Il backward layer procede a ritroso.
#
# Lo step legge 4 input:
# -v layer_file= i pesi del layer
# -v input_file= dal dataset o dall'output del layer precedente
# -v output_file= l'output calcolato nel forward pass
#
# Di questi solo uno, distinguendo se e' un hidden layer:
# [OUTPUT_LAYER] -v target_file= legge i target (in caso di ultimo layer)
# [HIDDEN_LAYER] -v next_delta_file= legge i delta del layer successivo
#

BEGIN {
    	# 1. Leggere i pesi del layer
	activation_function = read_layer_file(layer_file, neuron_matrix)

    	# 2. Leggere l'input (dal dataset o dall'output del layer precedente)
    	read_input_file(input_file, input_matrix, bias_mode)

    	# 3. Leggere l'output già calcolato nel forward pass
    	read_input_file(output_file, output_matrix, "none")

    	# 4. Leggere i target o i delta del layer successivo
    	if (is_output_layer) {
        	# Target (solo per l'ultimo layer)
        	read_target_values(target_file, target_matrix)
		if (debug) {
			printf("[DEBUG]: nnet-backward-layer: loaded target_matrix\n")
			print_matrix(target_matrix)
		}
    	} 
	else {
        	# Delta del layer successivo
        	read_input_file(next_delta_file, next_delta_matrix, "none")
    	}

    	# 5. Calcolo dei delta per ogni neurone

   	# Vediamo le dimensioni della matrice:
	num_inputs = input_matrix[0, 0]
	if (debug) printf("[DEBUG] nnet-backward-layer: num_inputs = %d\n", num_inputs)
	num_neurons = neuron_matrix[0, 0] 
	if (debug) printf("[DEBUG] nnet-backward-layer: num_neurons = %d\n", num_neurons)

	# Cicliamo sugli input:
    	for (row = 1; row <= num_inputs; row++) {
		# Calcoliamo i valori:
        	for (neuron = 1; neuron <= num_neurons; neuron++) {
            		# Delta diverso se siamo nell'output layer o hidden layer
            		if (is_output_layer) {
				# Estraiamo il valore target del neurone:
				target = target_matrix[row, neuron]
				if (debug) printf("[DEBUG] nnet-backward-layer: target_matrix[%d, %d] = %f\n", row, neuron, target)
				output = output_matrix[row, neuron]
				if (debug) printf("[DEBUG] nnet-backward-layer: output_matrix[%d, %d] = %f\n", row, neuron, output)
				
				# Calcolo dell'errore:
                		error = target - output
				if (debug) printf("[DEBUG] nnet-backward-layer: error (%f - %f) = %f\n", target, output, error)
            		} 
			# Siamo nell'hidden-layer:
			else {
				# Somma pesata per l'Hidden Layer, consideriamo il delta:
				error = sum_weighted_delta(neuron, next_neuron_matrix, next_delta_matrix)
				output = output_matrix[row, neuron]
            		}

			# Applicchiamo il delta:
			activation_derivative = apply_activation_derivative(output, activation_function)
			if (debug) printf("[DEBUG] nnet-backward-layer: activation_derivative = %f\n", activation_derivative)
	      		delta_value = error * activation_derivative
			if (debug) printf("[DEBUG] nnet-backward-layer: delta_value = %f\n", delta_value)

            		# Salviamo il delta
            		delta_matrix[row, neuron] = delta_value
			if (debug) printf("[DEBUG] nnet-backward-layer: delta_matrix[%d, %d] = %f\n", row, neuron, delta_matrix[row, neuron])
        	}
		# Dimensione colonna della delta_matrix:
		delta_matrix[row, 0] = num_neurons
    	}
	# Dimensione righe delta_matrix:
	delta_matrix[0, 0] = num_inputs
	# Salviamo su file la DELTA_MATRIX:
    	write_matrix(delta_output_file, delta_matrix)

    	# 6. Calcolo dei gradienti per aggiornare i pesi
	# Numero di colonne di input (input reali + eventuale bias)
	num_input_cols = input_matrix[1, 0]
    	for (neuron = 1; neuron <= num_neurons; neuron++) {
		# Cicliamo su ciascun input collegato al neurone
    		# (compreso il bias se presente)
    		for (col = 1; col <= num_input_cols; col++) {
			# Somma temporanea dei contributi dei delta * input per ogni pattern
        		sum = 0

        		# Cicliamo su ogni pattern di input (ogni riga)
        		for (row = 1; row <= num_inputs; row++) {
            			# delta_matrix[row, neuron] = delta del neurone "neuron"
            			# input_matrix[row, col]   = input j-esimo per il pattern row
            			sum += delta_matrix[row, neuron] * input_matrix[row, col]
        		}
			
			# Media sui pattern: il gradiente è la media dei contributi
        		grad = sum / num_inputs

        		# Salviamo il gradiente nella matrice
        		# gradient_matrix ha la stessa dimensione della matrice dei pesi
        		gradient_matrix[neuron, col] = grad
		}
		# Dimensione colonne riga, in gradient_matrix:
		gradient_matrix[neuron, 0] = num_input_cols
	}
	# Dimensione righe: gradient_matrix
	gradient_matrix[0, 0] = num_neurons	

    	# 7. Output su file: gradient_matrix
    	write_matrix(gradient_output_file, gradient_matrix)
}

