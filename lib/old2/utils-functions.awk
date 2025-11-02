#
# Libreria di funzioni di base e utility per leggere e scrivere file e caricare
# i file di weights con aggiunta di bias o meno.
#
# PARAMETRI:
# -v add_bias = 1 --> aggiunge un bias sovrascrivendo l'ultima colonna, in modo da usare il file di inputs.
#

# Funzione per rendere una stringa di valori, un array: 
function split_line_to_array(line, array,   i, a_length) {
	# Splittiamo la riga in array in base agli spazi:
	a_length = split(line, array, /[ \t]+/)
	for (i = 1; i <= a_length; i++) {
		array[i] += 0	# Conversione numerica forzata.
	}
	# Restituiamo la lunghezza array:
	array[0] = a_length
	return array[0]
}

# Funzione di debug, per stampa array:
function print_array(array,    i) {
	printf("[DEBUG] print_array: array_len=%d\n", array[0])
	for (i=1; i<=array[0]; i++) {
		printf("[DEBUG] print_array: element[%d]=%f\n", i, array[i])
	}
}

# Funzione di debug per stampa matrice:
function print_matrix(matrix,    row, nrows, col, ncols) {
	printf("[DEBUG] print_matrix: matrix [\n")
	nrows = matrix[0, 0]
	for (row=1; row<=nrows; row++) {
		ncols = matrix[row, 0]
		printf("[DEBUG] print_matrix: ");
		for (col=1; col<=ncols; col++) {
			printf("%f ", matrix[row, col])
		}
		printf("\n")
	}
	printf("[DEBUG] print_matrix: ] = size [%d x %d]\n", nrows, ncols) 
}

# Operazione su MATRICE: passiamo una matrice, un array e una riga e riempiamo
# la riga della matrice con quell'array.
# La convenzione e' che la cella 0 (colonna = 0) della riga riempita ospiti
# la quantita' di colonne della riga attuale.
function copy_array_to_matrix_row(matrix, m_row, array,    col) {
	# Cicliamo sull'array 
	for (col=1; col<=array[0]; col++) {
		matrix[m_row, col] = array[col]
	}
	# Inseriamo nella posizione 0 della riga il numero di colonne:
	matrix[m_row, 0] = array[0]
	return matrix[m_row, 0]
}    

# Operazione su MATRICE: estraiamo una riga di una matrice per poterla passare
# ad altre operazioni matematiche.
function copy_matrix_row_to_array(matrix, m_row, array,    col) {
	# Copiamo i valori di una riga nell'array:
	array[0] = matrix[m_row, 0]
	for (col=1; col<=array[0]; col++) {
		array[col] = matrix[m_row, col]
	}
	# Restituiamo lunghezza: 
	return array[0]
}

# Funzione per leggere solo l'ultima colonna di un file, serve nel BACKWARD-STEP
# quando siamo in output_layer, ci serve avere i valori target di input.
function read_target_values(file, matrix,    rows, line, len, splitted) {
	rows=0	

	# Leggiamo il file per righe:
	while ((getline line < file) > 0) {
		rows++
		len = split_line_to_array(line, splitted)
		if (debug_utils) printf("[DEBUG] read_target_values: line = %s, splitted in %d values\n", line, len)

		# Salvo in matrice:
		matrix[rows, 1] = splitted[len]
		if (debug_utils) printf("[DEBUG] read_target_values: matrix[%d, 1] = %f\n", rows, matrix[rows, 1]) 
		matrix[rows, 0] = 1
	}
	# Settiamo il numero di righe:
	matrix[0, 0] = rows
	if (debug_utils) printf("[DEBUG] read_target_values: rows of matrix = %d\n", matrix[0, 0])

	# Chiudo il file:
	close(file)	
}

# Funzione per leggere l'input file (con add_bias = 1, aggiunge 1.0 come bias)
# se add_bias = 0 serve per i layer successivi e lascia invariato il bias presente.
# utile per FORWARD_STEP.
# Il parametro BIAS_MODE:
# - "" o "none" -> non comporta modifiche alla matrice.
# - "replace"   -> sostituisce l'ultima colonna con 1.0
# - "append"    -> aggiunge una colonna con 1.0 alla fine.
function read_input_file(file, inputs, bias_mode,    line, i, rows, tmp_array, len) {
	if (debug_utils) {
		printf("[DEBUG] read_input_file: input_file = %s\n", file)
		printf("[DEBUG] read_input_file: bias_mode  = %s\n", bias_mode)
	}
	# Inizializziamo la rows:
	rows = 0
	
	# Leggiamo le righe del file:
	while ((getline line < file) > 0) {
		rows++
		len = split_line_to_array(line, tmp_array)
		if (debug_utils) {
			printf("[DEBUG] read_input_file: line = %s, splitted len = %d\n", line, len)
		}
	
		# Conversione del bias in 1.0 per input:	
		if (bias_mode == "replace") {
			# Sostituiamo un elemento:
			tmp_array[len] = 1.0	# Sostituzione target con bias = 1.0
			if (debug_utils) printf("[DEBUG] read_input_file: replacing bias = %d, in position %d\n", tmp_array[len], len)
		}
		else if (bias_mode == "append") {
			# Aggiungiamo un elemento:
			len++
			tmp_array[len] = 1.0	# Aggiunta target con bias = 1.0
			if (debug_utils) printf("[DEBUG] read_input_file: adding bias = %d, in position %d\n", tmp_array[len], len)
		}

		# Settiamo la dimensione dell'array:
		tmp_array[0] = len
		if (debug_utils) printf("[DEBUG] read_input_file: setting length of tmp_array = %d\n", tmp_array[0])
		
		# Inseriamo i valori nell'inputs come matrice:
		copy_array_to_matrix_row(inputs, rows, tmp_array)
		if (debug_utils) {
			print_array(tmp_array)
			printf("[DEBUG] read_input_file: copy array to matrix row = %d\n", rows) 
		}
	}
	# Inseriamo per le righe lo stesso dato: RIGA 0, COLONNA 0 = Numero righe di matrice.
	inputs[0, 0] = rows
	if (debug_utils) printf("[DEBUG] read_input_file: total number of matrix_rows = %d\n", rows)

	# Chiudiamo il file da cui abbiamo letto:
	close(file)
	# Restituisco numero di righe:
	return rows
}

# Funzione per leggere un layer file quindi deve omettere le righe con commento,
# le righe:
# - ACTIVATION=
# E mettere il tutto in una matrice, come sempre la convenzione per le matrici e funzioni
# e' che restituiamo nella funzione il numero di righe lette, e nella posizione 0 della riga
# il numero di colonne di quella specifica riga.
#
# Inoltre il layer file viene parsato e letta la funzione di attivazione, che viene restituita
# come parametro.
function read_layer_file(layer_file, layer,    rows, tmp_array, kv, activation_function) {
	rows = 0	

	# Stampa solo per debug:
	if (debug_utils) {
		printf("[DEBUG] read_layer_file: layer_file=%s\n", layer_file)
	}

	# Lettura per righe del file:
	while ((getline line < layer_file) > 0) {
		if (debug_utils) {
			printf("[DEBUG] read_layer_file: line=%s\n", line)
		}

		# Escludiamo i commenti:
		if (line ~ /^#/) continue

		# Leggiamo la activation function:
		if (line ~ /^ACTIVATION=/) {
			split(line, kv, "=")
			activation_function = kv[2]	
			if (debug_utils) {
				printf("[DEBUG] read_layer_file: activation_function=%s\n", activation_function)
			}
			continue
		}

		# Altrimenti leggiamo:
		split_line_to_array(line, tmp_array)
		rows++

		# Se debug, stampo array:
		if (debug_utils) {
			print_array(tmp_array)
		}

		# Inseriamo i valori letti:
		copy_array_to_matrix_row(layer, rows, tmp_array) 
	}
	# Inseriamo per le righe:
	layer[0, 0] = rows

	# Chiusura del file:
	close(layer_file)

	# Restituiamo il numero di righe lette:
	return activation_function
}


# Scriviamo la matrice sul file passato come parametro:
function write_matrix(output_file, matrix,    row, col, nrows, ncols) {
	# Sovrascrittura file:
	printf("") > output_file

	# Recupero numero righe:
	nrows = matrix[0, 0]
	for (row = 1; row <= nrows; row++) {
		# Recupero numero colonne:
		ncols = matrix[row, 0]
		for (col = 1; col <= ncols; col++) {
			# Stampo su file a seconda se sia ultima colonna o meno:
			if (col < ncols) printf("%f ", matrix[row, col]) >> output_file
			else printf("%f", matrix[row, col]) >> output_file
		}
		printf("\n") >> output_file
	}

	# Chiusura file:
	close(output_file)		
}

