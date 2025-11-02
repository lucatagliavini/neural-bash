#
# Lo script serve per aggiornare i pesi dei layer in questa modalita':
#
# Layer:
# W11, W12, B11
# W21, W22, B21
#
# Tutti gli elementi:
# ELEMENT = ELEMENT - (LEARNING_RATE * GRADIENT)
#
# Quindi allo script serve:
# - il layer file (in / out)
# - il gradient file (calcolo differenze)
# - il learning rate
#

BEGIN {
	if (learning_rate == 0) {
		learning_rate = 0.1
	}
	# Stampiamo:
	if (debug) {
		printf("[DEBUG] nnet-update-layer: learning_rate = %f\n", learning_rate)
	}

	# Carichiamo il layer file:
	activation_function = read_layer_file(layer_file, neuron_matrix)
	if (debug) {
		printf("[DEBUG] nnet-update-layer: layer_file=%s\n", layer_file)
		printf("[DEBUG] nnet-update-layer: activation_function=%s\n", activation_function)
		printf("[DEBUG] nnet-update-layer: neuron_matrix\n")
		print_matrix(neuron_matrix)
	}

	# Carichiamo il gradient file:
	read_layer_file(gradient_file, gradient_matrix)
	if (debug) {
		printf("[DEBUG] nnet-update-layer: gradient_file=%s\n", gradient_file)
		printf("[DEBUG] nnet-update-layer: gradient_matrix\n")
		print_matrix(gradient_matrix)
	}

	# A questo punto possiamo eseguire il ciclo di aggiornamento:
	num_neurons = neuron_matrix[0, 0]
	num_inputs = neuron_matrix[1, 0]
	if (debug) printf("[DEBUG] nnet-update-layer: updating neuron_matrix size [%d x %d]\n", num_neurons, num_inputs)

	# Aggiorniamo con learning rate:
	for (neuron=1; neuron<=num_neurons; neuron++) {
		for(input=1; input<=num_inputs; input++) {
			# Valore da aggiornare:
			old_value = neuron_matrix[neuron, input]
			gradient = gradient_matrix[neuron, input]
			# Aggiornamento:
			neuron_matrix[neuron, input] = old_value - (gradient * learning_rate)
		}
	}

	# Salviamo la matrice:
	if (debug) {
		printf("[DEBUG] nnet-update-layer: updated neuron_matrix")
		print_matrix(neuron_matrix)
	}

	# Output su file
}
