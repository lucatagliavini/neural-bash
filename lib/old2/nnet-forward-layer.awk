# ===============================================================
# Forward pass di un singolo layer
# ===============================================================
#
# Eseguiamo il forward step di un singolo layer, in caso facciamo
# il forward step del primo layer, dovremo impostare i bias a 1.0
# al posto dell'ultima colonna che sarebbero gli output del DS.
#
# In tutti gli altri casi non si considerano i bias, e si riportano
# quelli presenti in matrice.
#
# Parametri:
# -v layer_file=<path-to-layer-file>
# -v input_file=<path-to-input-file> 
# -v bias_mode=["none"|"replace"|"append"]
# -v output_file=<path-to-output-file>
#

# Blocco FORWARD_LAYER
BEGIN {
	# ----------------------------------------
    	# 1. Leggi i pesi dal file del layer
    	# ----------------------------------------
 	activation_function = read_layer_file(layer_file, neuron_matrix)

	# ----------------------------------------
	# 2. Leggi l'input (dataset o output layer precedente)
	# ----------------------------------------
    	read_input_file(input_file, input_matrix, bias_mode)

	# ----------------------------------------
	# 3. Calcola il forward pass per ogni riga di input
	# ----------------------------------------
	# Svuotiamo output file:
	printf("") > output_file
	
	# Carichiamo dimensioni delle matrici:
	num_inputs = input_matrix[0, 0]
	num_neurons = neuron_matrix[0, 0]

	# Ciclo su tutta la matrice inputs:
	for (row = 1; row <= num_inputs; row++) {
        	row_len = input_matrix[row, 0]

        	# Estrai input della riga corrente in un array temporaneo
		copy_matrix_row_to_array(input_matrix, row, input_row)

        	# Calcola l'output per ogni neurone del layer
        	for (neuron = 1; neuron <= num_neurons; neuron++) {
			# Estraggo riga di neuroni:
			copy_matrix_row_to_array(neuron_matrix, neuron, neuron_row)
	
			# Calcolo attivazione:
            		sum = dot_product(input_row, neuron_row)
            		out = apply_activation(sum, activation_function)

			# Output su FILE:
			if (neuron < num_neurons) printf("%f ", out) >> output_file
			else printf("%f", out) >> output_file
        	}
		# Separiamo la riga:
		printf("\n") >> output_file	
    	}

	# ----------------------------------------
	# 4. Chiudi l'output
	# ----------------------------------------
	close(output_file)
}
