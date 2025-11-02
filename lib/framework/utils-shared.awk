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
