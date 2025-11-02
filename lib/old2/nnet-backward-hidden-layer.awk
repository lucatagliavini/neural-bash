#
# Questo script awk, si propone di eseguire uno step di backpropagation in una struttura
# a layer che abbiamo implementato.
#
# Il backward layer procede a ritroso.
#
# Lo step legge gli input necessari per l'hidden layer
# la logica e' differente rispetto all'output layer, e si comporta,
# nel caso dello XOR:
#
# awk \
#  -v layer_file="models/xor/layer1.txt" \
#  -v input_file="dataset/xor.txt" \
#  -v output_file="tmp/xor-20250726/layer1.out" \
#  -v next_layer_file="models/xor/layer2.txt" \
#  -v next_delta_file="tmp/xor-20250726/layer2-delta.out" \
#  -v delta_output_file="tmp/xor-20250726/layer1-delta.out" \
#  -v gradient_output_file="tmp/xor-20250726/layer1-gradient.out" \
#  [-v bias_mode="replace"] \
#  [-v debug=1] \
#  -f lib/framework/utils-math.awk \
#  -f lib/framework/utils-activation.awk \
#  -f lib/framework/utils-functions.awk \
#  -f lib/framework/nnet-backward-hidden.awk \
#  /dev/null
#

BEGIN {
    	# 1. Leggere i pesi del layer
	activation_function = read_layer_file(layer_file, neuron_matrix)
	if (debug) {
        	printf("[DEBUG] nnet-backward-hidden-layer: loaded layer_file=%s\n", layer_file)
		printf("[DEBUG] nnet-backward-hidden-layer: activation_function = %s\n", activation_function)
		printf("[DEBUG] nnet-backward-hidden-layer: neuron_matrix\n")
		print_matrix(neuron_matrix)
	}

    	# 2. Leggere l'input (dal dataset o dall'output del layer precedente)
    	read_input_file(input_file, input_matrix, "append")
	if (debug) {
        	printf("[DEBUG] nnet-backward-hidden-layer: loaded input_file=%s\n", input_file)
		printf("[DEBUG] nnet-backward-hidden-layer: input_matrix\n")
		print_matrix(input_matrix)
	}

    	# 3. Leggere l'output già calcolato nel forward pass
    	read_input_file(output_file, output_matrix, "none")
	if (debug) {
                printf("[DEBUG] nnet-backward-hidden-layer: loaded output_file=%s\n", output_file)
                printf("[DEBUG] nnet-backward-hidden-layer: output_matrix\n")
                print_matrix(output_matrix)
        }

    	# 4a. Leggere i delta del layer successivo
       	read_input_file(next_delta_file, next_delta_matrix, "none")
	if (debug) {
                printf("[DEBUG] nnet-backward-hidden-layer: loaded next_delta_file=%s\n", next_delta_file)
		printf("[DEBUG] nnet-backward-hidden-layer: next_delta_matrix\n")
		print_matrix(next_delta_matrix)
	}

	# 4b. Leggere il next_layer : (ignoriamo la funzione di attivazione)
	read_layer_file(next_layer_file, next_layer_matrix)
	if (debug) {
                printf("[DEBUG] nnet-backward-hidden-layer: loaded next_layer_file=%s\n", next_layer_file)
                printf("[DEBUG] nnet-backward-hidden-layer: next_layer_matrix\n")
                print_matrix(next_layer_matrix)
        }

    	# 5. Calcolo dei delta per ogni neurone

   	# Vediamo le dimensioni della matrice:
	num_patterns = output_matrix[0, 0] 		# Righe della output matrix (layer corrente)
	if (debug) printf("[DEBUG] nnet-backward-hidden-layer: num_patterns = %d\n", num_patterns)
	num_neurons = output_matrix[1, 0]		# Colonne della output matrix 	
	if (debug) printf("[DEBUG] nnet-backward-hidden-layer: num_neurons_current = %d\n", num_neurons)
	num_neurons_next = next_layer_matrix[0, 0]	# Colonne della matrice del next-layer
	if (debug) printf("[DEBUG] nnet-backward-hidden-layer: num_neurons_next = %d\n", num_neurons_next)
	
	# I pattern:
	for (pattern = 1; pattern <= num_patterns; pattern++) {
		# Iniziamo il ciclo:
		if (debug) printf("[DEBUG] nnet-backward-hidden-layer: pattern = %d\n", pattern)
		
		# Sul numero neuroni del current layer:
		for (neuron = 1; neuron <= num_neurons; neuron++) {
			if (debug) printf("[DEBUG] nnet-backward-hidden-layer: neurone = %d/%d\n", neuron, num_neurons)
			# Calcolo somma pesata:
			sum = 0
			for (neuron_n = 1; neuron_n <= num_neurons_next; neuron_n++) {
				# Peso legato al neurone corrente:
				peso = next_layer_matrix[neuron_n, neuron]
				# Delta del neurone successivo
				delta_next = next_delta_matrix[pattern, neuron_n]
				
				# Somma:
				sum += peso * delta_next

				# Debug:
				if (debug) {
					printf("[DEBUG] nnet-backward-hidden-layer: neuron_next = %d -> peso[%d, %d]=%f * delta_next = %f\n", neuron_n, neuron_n, neuron, peso, delta_next)
				}
			}
			printf("[DEBUG] nnet-backward-hidden-layer: sum = %f\n", sum)
			
			# Derivata funzione di attivazione sul neurone corrente:
			y_value = output_matrix[pattern, neuron]
			derivate = apply_activation_derivative(y_value, activation_function)

			# Settaggio delta:
			delta_matrix[pattern, neuron] = sum * derivate
			if (debug) {
				printf("[DEBUG] nnet-backward-hidden-layer: output=%f, derivate=%f, delta_matrix[%d, %d]=%f\n", y_value, derivate, pattern, neuron, delta_matrix[pattern, neuron])
			}
		}
		# Settiamo le dimensioni colonna:
		delta_matrix[pattern, 0] = num_neurons
	}
	# Dimensione righe delta_matrix:
	delta_matrix[0, 0] = num_patterns
	# Salviamo su file la DELTA_MATRIX:
    	write_matrix(delta_output_file, delta_matrix)

	# 6. Calcolo dei Gradienti:
	num_inputs = input_matrix[1, 0] - 1
	init_matrix_zero(gradient_matrix, num_neurons, num_inputs)
	
	# Costruzione incrementale matrice:
	for (pattern = 1; pattern<=num_patterns; pattern++) {
		for (neuron = 1; neuron<=num_neurons; neuron++) {
			sum=0
			# Cicliamo:
			for (input = 1; input<=num_inputs; input++) {
				contributo = delta_matrix[pattern, neuron] * input_matrix[pattern, input]
				# Sommatoria sulla gradient:
				gradient_matrix[neuron, input] += contributo
			} 
			# Dimensione colonna gradient:
			gradient_matrix[neuron, 0] = num_inputs
		}
	}
	gradient_matrix[0, 0] = num_neurons	

	# Mediare pattern:
	for (neuron=1; neuron<=num_neurons; neuron++) {
		for(input=1; input<=num_inputs; input++) {
			gradient_matrix[neuron, input] /= num_patterns
		}
	}
	# Dimensione righe: gradient_matrix
    	write_matrix(gradient_output_file, gradient_matrix)

	# Stampa:
	if (debug) {
		printf("[DEBUG] nnet-backward-hidden-layer: gradient_matrix\n")
		print_matrix(gradient_matrix)
	}
}

