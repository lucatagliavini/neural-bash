#
# File di funzioni per la NNETWORK in memory.
#
# Tutte le matrici che andremo a fare, array multidimensionali,
# avranno:
#
# CONVENZIONE:
# ------------
# matrix[0, 0] = numero_righe
# matrix[i, 0] = numero_colonne della riga i-esima.
#
# array[0] = numero_elementi del vettore.
# 
# Tutti i cicli andranno da elemento 1 a elemento = a length.
#

# Funzione di logging, senza parametro vuol dire stdout,
# altrimenti puo' ridirigere su file.
function logmesg(flag, text, output) {
	# In mancanza del flag di debug, non sampa nulla
	if (flag == "" || flag == 0) return

	# Se abbiamo un file di output:
	if (output != "") {
		printf("%s", text) >> output
		close(output)
		return
	}

	# Altrimenti stampo su stderr:
	printf("%s", text) > "/dev/stderr"
}

# Funzione per loggare una matrice:
function logmatrix(flag, matrix, output,    nrows, row, ncols, col) {
	# Togliamo il log immediatamente:
	if (flag == "" || flag == 0) return
	
	# Dimensione della matrice:
	nrows = matrix[0, 0]
	ncols = matrix[1, 0]
	logmesg(flag, "Matrix (" nrows "x" ncols ") = [\n", output)

	# Cicliamo sulla matrice:
	for (row=1; row<=nrows; row++) {
		for (col=1; col<=ncols; col++) {
			# Distinguo caso intermedio e fine riga:
			if (col < ncols) logmesg(flag, matrix[row, col]" ", output)
			else logmesg(flag, matrix[row, col]"\n", output)
		}
	}
	# Chiudo la matrice:
	logmesg(flag, "]\n", output)
}

# Funzione per stampare una matrice dei layer dei pesi:
function logmatrix_weights(flag, layer_weights, layer_id, output,    nrows, row, ncols, col) {
	# Togliamo il log immediatamente:
        if (flag == "" || flag == 0) return

	# Dimensione della matrice:
	nrows = layer_weights[layer_id, 0, 0]
	ncols = layer_weights[layer_id, 1, 0]
	logmesg(flag, "Weights matrix for layer: "layer_id" ("nrows"x"ncols") = [\n", output)

	# Cicliamo sulla matrice:
	for (row=1; row<=nrows; row++) {
                for (col=1; col<=ncols; col++) {
                        # Distinguo caso intermedio e fine riga:
                        if (col < ncols) logmesg(flag, layer_weights[layer_id, row, col]" ", output)
                        else logmesg(flag, layer_weights[layer_id, row, col]"\n", output)
                }
        }
        # Chiudo la matrice:
        logmesg(flag, "]\n", output)
}

# Funzione per convertire una stringa in un vettore numerico:
function split_line_to_array(line, array,    i, len) {
	# Split per tutti i caratteri spazio, anche piu' di uno:
	len = split(line, array, /[ \t]+/)
	# Esplicita conversione in numero:
	for (i=1; i<=len; i++) {
		array[i] += 0
	}
	# Restituisco la lunghezza:
	array[0] = len
	return array[0]
}

# Funzione per convertire un array in stringa:
function array_to_string(array,    i, str) {
	# Appendo alla stringa:
	str = "["
	for (i=1; i<=array[0]; i++) {
		if (i < array[0]) str = str array[i] ", "
		else str = str array[i]
	}
	str = str"]"
	return str
}

# Funzione per aggiungere la riga a una matrice:
function copy_array_to_matrix_row(matrix, row_index, row_array,    col) {
	# Cicliamo sugli elementi per copiarli:
	for (col=1; col<=row_array[0]; col++) {
		matrix[row_index, col] = row_array[col]
	}
	# Impostiamo lunghezza:
	matrix[row_index, 0] = row_array[0]
	# Restituisco elementi copiati:
	return matrix[row_index, 0]
}

# Funzione quando aggiungiamo una riga di pesi a una matrice di layer:
function copy_weights_to_layer_matrix_row(layer_id, layer_weights, row, weights_array,    col) {
	# Cicliamo sugli elementi per copiarli:
	for (col = 1; col<=weights_array[0]; col++) {
		# La layer matrix contiene i pesi di tutti i layer:
		layer_weights[layer_id, row, col] = weights_array[col]
	}
	# Impostiamo la dimensione:
	layer_weights[layer_id, row, 0] = weights_array[0]
	return layer_weights[layer_id, row, 0]
}

# Funzione per aggiungere la riga fino ad indice fornito alla matrice:
function copy_array_to_matrix_row_with_index(matrix, row_index, row_array, from_index, to_index,    col, mcol) {
	# Cicliamo sugli elementi per copiarli:
	mcol=1
	for (col=from_index; col<=to_index; col++) {
		matrix[row_index, mcol] = row_array[col]
		mcol++
	}
	# Lunghezza: comprensiva di finali:
	matrix[row_index, 0] = (to_index - from_index + 1) 
	return matrix[row_index, 0]
}

# Funzione per copiare una riga di matrice in un array:
function copy_matrix_row_to_array(matrix, row_index, array,    ncols, col) {
	ncols = matrix[row_index, 0]
	# Cicliamo sugli elementi della colonna:
	for (col=1; col<=ncols; col++) {
		# Copio l'elemento nel vettore:
		array[col] = matrix[row_index, col]
	}
	# Restituisco elementi copiati:
	array[0] = ncols
	return array[0]
}

# Load dei layers della rete:
function load_layers(model_dir, num_layers, layer_meta, layer_weights,    layer_id, layer_file) {
	# Dobbiamo procedere a caricare tutti i file presenti nella cartella:
	for (layer_id = 1; layer_id<=num_layers; layer_id++) {
		# Carichiamo i layer uno alla volta:
		layer_file = model_dir"/layer"layer_id".txt"
		
		# Carichiamo:
		load_layer(layer_file, layer_id, layer_meta, layer_weights)
	}
	# Numero di layers:
	layer_meta[0, 0, 0] = num_layers
}

# Carica un singolo layer:
function load_layer(layer_file, layer_id, layer_meta, layer_weights,    nrow, nrow_meta, line, line_array, ncol, kv, activation_function) {
	nrow=0
	nrow_meta = 0

	# Apriamo il file e lo carichiamo per righe:
	while((getline line < layer_file) > 0) {
		# Saltiamo commenti:
		if (line ~ /^#/ || line ~ /^[[:space:]]*$/) continue

		# Se abbiamo la funzione di attivazione:
		if (line ~ /^ACTIVATION=/ ) {
			split(line, kv, "=")
                        activation_function = kv[2]
			
			# Aggiungo dato alla meta:
			nrow_meta++
			layer_meta[layer_id, "activation"] = activation_function
			layer_meta[layer_id, 1, 0] = 1 

			# Saltiamo il resto:
			continue
		}

		# Ora aggiungiamo una riga:
		ncol = split_line_to_array(line, line_array)
		nrow++

		# Aggiorniamo il metadato:
		if (nrow == 1) {
			layer_meta[layer_id, "num_inputs"] = ncol
			layer_meta[layer_id, 2, 0] = 1
		}	
		
		# Inseriamo nella matrice dei pesi:
		copy_weights_to_layer_matrix_row(layer_id, layer_weights, nrow, line_array)
	}
	# Aggiorniamo il layer con i dati dei neuroni:
	layer_meta[layer_id, "num_neurons"] = nrow
	layer_meta[layer_id, 3, 0] = 1
	# Inseriamo il layer (HAS_BIAS = TRUE)
	layer_meta[layer_id, "has_bias"] = 1
	layer_meta[layer_id, 4, 0] = 1
	# Numero totale di righe del meta layer: 4
	layer_meta[layer_id, 0, 0] = 4

	# Salviamo dimensione della layer_weighs:
	layer_weights[layer_id, 0, 0] = nrow

	# Chiusura del file:
	close(layer_file)
}

# Load del dataset che suddivide il tutto in matrici:
function load_dataset(dataset_file, num_inputs, dataset_meta, dataset_weights, dataset_targets,    line, line_array, nrow, ncol, bias_col, num_outputs) {
	# Ci servira' poi come dato:
	nrow=0
	num_outputs=0
	bias_col = num_inputs +1

	# Apriamo il file e leggiamo una riga alla volta:
	while((getline line < dataset_file) > 0) {
		# Ignoriamo le righe con commenti:
		if (line ~ /^#/ || line ~ /^[[:space:]]*$/) continue

		# C'é una nuova riga:
		ncol = split_line_to_array(line, line_array)
		nrow++

		# Salviamo num outputs:
		if (nrow == 1) num_outputs = ncol - num_inputs
		
		# Aggiunta della riga alla matrice dataset_weights, e aggiunta bias:
		copy_array_to_matrix_row_with_index(dataset_weights, nrow, line_array, 1, num_inputs)
		
		# Aggiunta della parte bias (con aggiornamento indice colonne)
		dataset_weights[nrow, bias_col] = 1.0
		dataset_weights[nrow, 0] = bias_col  
		
		# Salviamo la parte dei targets:
		copy_array_to_matrix_row_with_index(dataset_targets, nrow, line_array, bias_col, ncol)		
	}
	# Impostiamo dimensione totale di dataset_weights, e dataset_targets:
	dataset_weights[0, 0] = nrow
	dataset_targets[0, 0] = nrow

	# Impostazione metadati:
	dataset_meta["num_samples"] = nrow
	dataset_meta[1, 0] = 1
	dataset_meta["num_inputs"] = num_inputs 
	dataset_meta[2, 0] = 1
	dataset_meta["num_outputs"] = num_outputs 
	dataset_meta[3, 0] = 1
	dataset_meta[0, 0] = 3

	# Ultimo step: chiusura file:
	close(dataset_file)	
}

# ===============================================================================================
# FUNZIONI HELPER PER METADATI
# ===============================================================================================

# Estrae metadati comuni del dataset in un array associativo
# Uso: 	get_dataset_info(dataset_meta, info)
# 	Poi si accede con: info["num_samples"], info["num_inputs"], info["num_outputs"]
function get_dataset_info(dataset_meta, info) {
	info["num_samples"] = dataset_meta["num_samples"]
	info["num_inputs"] = dataset_meta["num_inputs"]
	info["num_outputs"] = dataset_meta["num_outputs"]
}


# Estrae metadati di un layer specifico
# Uso:	get_layer_info(layer_meta, 1, info)
# 	Poi accedi con: info["num_neurons"], info["activation"], ecc...
function get_layer_info(layer_meta, layer_id, info) {
	info["num_neurons"] = layer_meta[layer_id, "num_neurons"]
	info["num_inputs"] = layer_meta[layer_id, "num_inputs"]
	info["activation"] = layer_meta[layer_id, "activation"]
	info["has_bias"] = layer_meta[layer_id, "has_bias"]
}


# Restituisce il numero totale di layer della rete:
# Uso: num_layers = get_num_layers(layer_meta)
function get_num_layers(layer_meta) {
	return layer_meta[0, 0, 0]
}

# ===============================================================================================
# FUNZIONI HELPER PER FORWARD PASS
# ===============================================================================================

# Prepara l'array di input per il prossimo layer:
# - Copia gli output del layer corrente
# - Aggiunge bias se necessario
# Restituisce il numero di elementi dell'array
function prepare_next_layer_input(layer_output, layer_id, sample, layer_meta, input_array,
				num_neurons, neuron, bias_index) {
	
	num_neurons = layer_meta[layer_id, "num_neurons"]

	# Puliamo array precedente:
	delete input_array

	# Copia output come input per il prossimo layer:
	for (neuron=1; neuron<=num_neurons; neuron++) {
		input_array[neuron] = layer_output[layer_id, sample, neuron]
	}

	# Se richiesto dal layer aggiungiamo BIAS:
	if (layer_meta[layer_id, "has_bias"]) {
		bias_index = num_neurons + 1
		input_array[bias_index] = 1.0
		input_array[0] = bias_index
	}
	else {
		input_array[0] = num_neurons
	}

	# Restituisco il numero di elementi:
	return input_array[0]
}

# ===============================================================================================
# FUNZIONI DI VALIDAZIONE
# ===============================================================================================

# Valida il nome della funzione di attivazione
# Restituisce 1 se valida, 0 se non valida (con messaggio di errore)
function validate_activation(activation) {
	if (	activation == "sigmoid" || activation == "tanh" ||
		activation == "relu" || activation == "leaky_relu") {
		return 1
	}
	printf("[ERROR] Invalid activation: %s\n", activation) > "/dev/stderr"
	printf("        Available: sigmoid, tanh, relu, leaky_relu\n") > "/dev/stderr"
	return 0
}


# Validiamo il metodo di inizializzazione dei pesi
# Restituisce 1 se valido, 0 se non valido (con messaggio di errore)
function validate_init_method(method) {
	if (method == "xavier" || method == "he" || method == "random") {
		return 1
	}
	printf("[ERROR] Invalid init_method: %s\n", method) > "/dev/stderr"
	printf("        Available: xavier, he, random\n") > "/dev/stderr"
	return 0
}


# Validiamo la funzione di loss
# Restituisce 1 se valida, 0 se non valida (con messaggio di errore)
function validate_loss_function(loss_function) {
	if (loss_function == "mse" || loss_function == "ce") {
		return 1
	}
	printf("[ERROR] Invalid loss function: %s\n", loss_function) > "/dev/stderr"
	printf("        Available: mse, ce\n") > "/dev/stderr"
	return 0
}

