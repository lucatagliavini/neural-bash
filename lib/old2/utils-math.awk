#
# Gruppo di funzioni di utility per eseguire operazioni matriciali.
#
# Le matrici sono nella seguente convenzione:
# matrix[0, 0] = numero_righe_totali_matrice
# matrix[row, 0] = numero_colonne_riga_attuale
#
# sono caricate secondo questo standard dal layer e dai file numerici.
#

# Eseguiamo DOT_PRODUCT di vettore x matrix_row:
function dot_product(vector, matrix_row,    sum, i) {
	sum = 0
	# Calcoliamo il dot_product:
	for (i=1; i<=vector[0]; i++) {
		sum += vector[i] * matrix_row[i]

		# Debug:
		if (debug_math) {
			printf("[DEBUG] dot_product i=%d, vector=%f, weight=%f, partial_sum=%f\n",
			i, vector[i], matrix_row[i], sum) > "/dev/stderr"
		}
	}
	return sum
}

# Inizializziamo una matrice vuota:
function init_matrix_zero(matrix, nrows, ncols,    row, col) {
	for (row = 1; row<=nrows; row++) {
		for(col=1; col<=ncols; col++) {
			matrix[row, col] = 0
		}
		matrix[row, 0] = ncols
	}
	matrix[0, 0] = nrows
}

# Somma pesata (non media):
# neuron_index = neurone dell'hidden layer su cui stiamo calcolando il delta
# next_weights = matrice dei pesi del layer successivo
# next_deltas  = delta del layer successivo (per pattern specifico)
function sum_weighted_delta(neuron_index, next_weights, next_deltas,    k, sum) {
	sum = 0
	# Ignoriamo l'ultima colonna della riga, che è il peso del bias
	for (k = 1; k <= next_weights[0,0]; k++) {
		sum += next_weights[k, neuron_index] * next_deltas[k]
	}
	# Somma pesata:
	return sum
}

