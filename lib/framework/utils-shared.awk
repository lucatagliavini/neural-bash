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

# Svuota un intero array. Sostituisce "delete array" (non POSIX prima di POSIX.1-2008).
function clear_array(arr,    k) { for (k in arr) delete arr[k] }

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
function load_layer(layer_file, layer_id, layer_meta, layer_weights,    nrow, nrow_meta, line, line_array, ncol, kv, act_parts, activation_function, alpha) {
	nrow=0
	nrow_meta = 0

	# Apriamo il file e lo carichiamo per righe:
	while((getline line < layer_file) > 0) {
		# Saltiamo commenti:
		if (line ~ /^#/ || line ~ /^[[:space:]]*$/) continue

		# Se abbiamo la funzione di attivazione:
		if (line ~ /^ACTIVATION=/ ) {
			split(line, kv, "=")

			# Separiamo "leaky_relu:0.1" in nome e alpha (default 0.01)
			split(kv[2], act_parts, ":")
			activation_function = act_parts[1]
			alpha = (act_parts[2] != "" ? act_parts[2]+0 : 0.01)

			layer_meta[layer_id, "activation"] = activation_function
			layer_meta[layer_id, "alpha"]      = alpha
			continue
		}

		# Ora aggiungiamo una riga:
		ncol = split_line_to_array(line, line_array)
		nrow++

		if (nrow == 1) layer_meta[layer_id, "num_inputs"] = ncol

		# Inseriamo nella matrice dei pesi:
		copy_weights_to_layer_matrix_row(layer_id, layer_weights, nrow, line_array)
	}
	layer_meta[layer_id, "num_neurons"] = nrow
	layer_meta[layer_id, "has_bias"]    = 1

	# Salviamo dimensione della layer_weighs:
	layer_weights[layer_id, 0, 0] = nrow

	# Chiusura del file:
	close(layer_file)
}

# Load del dataset con opzionale train/val split.
# val_split in [0, 1): frazione dei campioni da riservare alla validazione.
# Se val_split <= 0 tutti i campioni vanno in train e val_* restano vuoti.
function load_dataset(dataset_file, num_inputs, dataset_meta, dataset_weights, dataset_targets,
                      val_weights, val_targets, val_split,
                      line, line_array, nrow, ncol, bias_col, num_outputs,
                      raw_w, raw_t, idx, i, j, tmp_i, tmp_j, n_val, n_train, src) {
	nrow=0
	num_outputs=0
	bias_col = num_inputs + 1

	# Leggiamo tutti i campioni in array temporanei indicizzati 1..N
	while ((getline line < dataset_file) > 0) {
		if (line ~ /^#/ || line ~ /^[[:space:]]*$/) continue
		ncol = split_line_to_array(line, line_array)
		nrow++
		if (nrow == 1) num_outputs = ncol - num_inputs

		# input + bias
		copy_array_to_matrix_row_with_index(raw_w, nrow, line_array, 1, num_inputs)
		raw_w[nrow, bias_col] = 1.0
		raw_w[nrow, 0]        = bias_col

		# target
		copy_array_to_matrix_row_with_index(raw_t, nrow, line_array, bias_col, ncol)
	}
	close(dataset_file)

	# Costruiamo un array di indici [1..nrow] e lo mescoliamo con Fisher-Yates
	for (i = 1; i <= nrow; i++) idx[i] = i
	for (i = nrow; i >= 2; i--) {
		j = int(rand() * i) + 1
		tmp_i = idx[i]; idx[i] = idx[j]; idx[j] = tmp_i
	}

	# Calcolo dimensioni split
	n_val   = (val_split > 0 && val_split < 1) ? int(nrow * val_split + 0.5) : 0
	if (n_val >= nrow) n_val = nrow - 1
	n_train = nrow - n_val

	# Copia nei set finali
	for (i = 1; i <= n_train; i++) {
		src = idx[i]
		for (j = 0; j <= raw_w[src, 0]; j++) dataset_weights[i, j] = raw_w[src, j]
		for (j = 0; j <= num_outputs;   j++) dataset_targets[i, j]  = raw_t[src, j]
	}
	dataset_weights[0, 0] = n_train
	dataset_targets[0, 0] = n_train

	for (i = 1; i <= n_val; i++) {
		src = idx[n_train + i]
		for (j = 0; j <= raw_w[src, 0]; j++) val_weights[i, j] = raw_w[src, j]
		for (j = 0; j <= num_outputs;   j++) val_targets[i, j]  = raw_t[src, j]
	}
	val_weights[0, 0] = n_val
	val_targets[0, 0] = n_val

	dataset_meta["num_samples"]     = n_train
	dataset_meta["num_val_samples"] = n_val
	dataset_meta["num_inputs"]      = num_inputs
	dataset_meta["num_outputs"]     = num_outputs
}

# Calcola media e deviazione standard per ogni feature (colonne 1..num_inputs).
# Salva in norm_stats["mean", f] e norm_stats["std", f].
# Usa SOLO i campioni del training set (dataset_weights, n campioni = dataset_weights[0,0]).
function compute_norm_stats(dataset_weights, num_inputs, norm_stats,
                            n, f, s, mean, variance, std,    i) {
    n = dataset_weights[0, 0]
    for (f = 1; f <= num_inputs; f++) {
        s = 0
        for (i = 1; i <= n; i++) s += dataset_weights[i, f]
        mean = s / n

        variance = 0
        for (i = 1; i <= n; i++) variance += (dataset_weights[i, f] - mean)^2
        variance /= n
        std = (variance > 0) ? sqrt(variance) : 1.0

        norm_stats["mean", f] = mean
        norm_stats["std",  f] = std
    }
    norm_stats["num_inputs"] = num_inputs
}

# Applica z-score in-place a tutti i campioni di una matrice dataset_weights.
# norm_stats deve essere già popolato da compute_norm_stats().
function apply_normalization(dataset_weights, norm_stats,    n, num_inputs, i, f) {
    n          = dataset_weights[0, 0]
    num_inputs = norm_stats["num_inputs"]
    for (i = 1; i <= n; i++)
        for (f = 1; f <= num_inputs; f++)
            dataset_weights[i, f] = (dataset_weights[i, f] - norm_stats["mean", f]) / norm_stats["std", f]
}

# Scrive le stats di normalizzazione su file (model_dir/normalize.conf).
function save_norm_stats(model_dir, norm_stats,    f, num_inputs, outfile) {
    num_inputs = norm_stats["num_inputs"]
    outfile    = model_dir "/normalize.conf"
    printf("num_inputs=%d\n", num_inputs) > outfile
    for (f = 1; f <= num_inputs; f++)
        printf("mean_%d=%.10g\nstd_%d=%.10g\n", f, norm_stats["mean", f], f, norm_stats["std", f]) >> outfile
    close(outfile)
}

# Carica le stats di normalizzazione da model_dir/normalize.conf.
# Restituisce 1 se il file esiste, 0 altrimenti.
function load_norm_stats(model_dir, norm_stats,    infile, line, key, val, parts) {
    infile = model_dir "/normalize.conf"
    if ((getline line < infile) <= 0) { close(infile); return 0 }
    # Prima riga: num_inputs=N
    split(line, parts, "="); norm_stats["num_inputs"] = parts[2] + 0
    while ((getline line < infile) > 0) {
        split(line, parts, "=")
        key = parts[1]; val = parts[2] + 0
        if      (key ~ /^mean_/) { sub(/^mean_/, "", key); norm_stats["mean", key+0] = val }
        else if (key ~ /^std_/)  { sub(/^std_/,  "", key); norm_stats["std",  key+0] = val }
    }
    close(infile)
    return 1
}

function print_predictions(dataset_meta, dataset_targets, layer_meta, layer_output,
				num_samples, num_outputs, num_layers, sample, neuron,
				pred, target, correct, threshold, pred_str, target_str, status_str) {
	num_samples = dataset_meta["num_samples"]
	num_outputs = dataset_meta["num_outputs"]
	num_layers  = layer_meta[0, 0, 0]
	threshold   = 0.5

	printf("================================================================================\n")
	printf("PREDICTIONS\n")
	printf("================================================================================\n")

	if (task == "regression") {
		printf("%-8s | %-12s | %-12s | %-12s\n", "Sample", "Predicted", "Target", "Abs.Error")
		printf("--------------------------------------------------------------------------------\n")
		for (sample = 1; sample <= num_samples; sample++) {
			pred_str = ""; target_str = ""
			for (neuron = 1; neuron <= num_outputs; neuron++) {
				pred   = layer_output[num_layers, sample, neuron]
				target = dataset_targets[sample, neuron]
				pred_str   = pred_str   (neuron == 1 ? "" : " ") sprintf("%.6f", pred)
				target_str = target_str (neuron == 1 ? "" : " ") sprintf("%.6f", target)
			}
			abs_err = (pred - target < 0) ? -(pred - target) : (pred - target)
			printf("%-8d | %-12s | %-12s | %-12.6f\n", sample, pred_str, target_str, abs_err)
		}
	} else {
		printf("%-8s | %-20s | %-15s | %-10s\n", "Sample", "Predicted", "Target", "Status")
		printf("--------------------------------------------------------------------------------\n")
		for (sample = 1; sample <= num_samples; sample++) {
			pred_str = ""
			for (neuron = 1; neuron <= num_outputs; neuron++) {
				pred = layer_output[num_layers, sample, neuron]
				pred_str = pred_str (neuron == 1 ? "" : " ") sprintf("%.4f", pred)
			}
			target_str = ""
			for (neuron = 1; neuron <= num_outputs; neuron++) {
				target = dataset_targets[sample, neuron]
				target_str = target_str (neuron == 1 ? "" : " ") sprintf("%s", target)
			}
			if (num_outputs > 1) {
				pred_class = 1; target_class = 1
				for (neuron = 2; neuron <= num_outputs; neuron++) {
					if (layer_output[num_layers, sample, neuron] > layer_output[num_layers, sample, pred_class])
						pred_class = neuron
					if (dataset_targets[sample, neuron] > dataset_targets[sample, target_class])
						target_class = neuron
				}
				correct = (pred_class == target_class)
			} else {
				pred = layer_output[num_layers, sample, 1]
				target = dataset_targets[sample, 1]
				correct = ((pred >= threshold ? 1 : 0) == target)
			}
			status_str = (correct ? "✓ CORRECT" : "✗ WRONG")
			printf("%-8d | %-20s | %-15s | %-10s\n", sample, pred_str, target_str, status_str)
		}
	}
	printf("================================================================================\n")
	printf("\n")
}

# Stampa la confusion matrix per classificazione.
# Per binaria (num_outputs=1): mappa pred/target {0,1} → classi {1,2}.
# Per multi-classe (num_outputs>1): usa argmax.
function print_confusion_matrix(dataset_meta, dataset_targets, layer_meta, layer_output,
			num_samples, num_outputs, num_layers, threshold,
			sample, neuron, pred_class, target_class, n_classes,
			cm, i, j, col_w, row_sum, label) {
	num_samples = dataset_meta["num_samples"]
	num_outputs = dataset_meta["num_outputs"]
	num_layers  = layer_meta[0, 0, 0]
	threshold   = 0.5

	n_classes = (num_outputs == 1) ? 2 : num_outputs

	# Azzera matrice
	for (i = 1; i <= n_classes; i++)
		for (j = 1; j <= n_classes; j++)
			cm[i, j] = 0

	for (sample = 1; sample <= num_samples; sample++) {
		if (num_outputs == 1) {
			target_class = (dataset_targets[sample, 1] >= threshold) ? 2 : 1
			pred_class   = (layer_output[num_layers, sample, 1] >= threshold) ? 2 : 1
		} else {
			pred_class = 1; target_class = 1
			for (neuron = 2; neuron <= num_outputs; neuron++) {
				if (layer_output[num_layers, sample, neuron] > layer_output[num_layers, sample, pred_class])
					pred_class = neuron
				if (dataset_targets[sample, neuron] > dataset_targets[sample, target_class])
					target_class = neuron
			}
		}
		cm[target_class, pred_class]++
	}

	col_w = 8
	printf("\nCONFUSION MATRIX  (rows=actual, cols=predicted)\n")
	printf("%-8s", "")
	for (j = 1; j <= n_classes; j++) printf("  %-*s", col_w, "C" j)
	printf("\n")
	for (i = 1; i <= n_classes; i++) {
		printf("%-8s", "C" i)
		for (j = 1; j <= n_classes; j++) printf("  %-*d", col_w, cm[i, j])
		printf("\n")
	}
	printf("\n")
}

function print_metrics(dataset_meta, dataset_targets, layer_meta, layer_output,
			num_samples, num_outputs, num_layers, mse, mae, accuracy,
			sample, neuron, pred, target, correct, total_correct, threshold,
			pred_class, target_class, sum_abs) {
	num_samples = dataset_meta["num_samples"]
	num_outputs = dataset_meta["num_outputs"]
	num_layers  = layer_meta[0, 0, 0]
	threshold   = 0.5

	mse = compute_mse(dataset_meta, dataset_targets, layer_meta, layer_output)

	# MAE — utile sia per regressione sia come metrica aggiuntiva in classificazione
	sum_abs = 0
	for (sample = 1; sample <= num_samples; sample++)
		for (neuron = 1; neuron <= num_outputs; neuron++)
			sum_abs += (layer_output[num_layers, sample, neuron] - dataset_targets[sample, neuron]) < 0 \
			           ? -(layer_output[num_layers, sample, neuron] - dataset_targets[sample, neuron]) \
			           :  (layer_output[num_layers, sample, neuron] - dataset_targets[sample, neuron])
	mae = sum_abs / (num_samples * num_outputs)

	printf("EVALUATION METRICS\n")
	printf("================================================================================\n")
	printf("Mean Squared Error (MSE)  : %.6f\n", mse)
	printf("Mean Absolute Error (MAE) : %.6f\n", mae)

	if (task == "regression") {
		printf("================================================================================\n")
		return
	}

	# Classificazione: accuracy + confusion matrix
	total_correct = 0
	for (sample = 1; sample <= num_samples; sample++) {
		if (num_outputs > 1) {
			pred_class = 1; target_class = 1
			for (neuron = 2; neuron <= num_outputs; neuron++) {
				if (layer_output[num_layers, sample, neuron] > layer_output[num_layers, sample, pred_class])
					pred_class = neuron
				if (dataset_targets[sample, neuron] > dataset_targets[sample, target_class])
					target_class = neuron
			}
			if (pred_class == target_class) total_correct++
		} else {
			pred   = layer_output[num_layers, sample, 1]
			target = dataset_targets[sample, 1]
			if ((pred >= threshold ? 1 : 0) == target) total_correct++
		}
	}
	accuracy = (total_correct / num_samples) * 100

	printf("Accuracy                  : %.2f%% (%d/%d)\n", accuracy, total_correct, num_samples)
	printf("================================================================================\n")

	print_confusion_matrix(dataset_meta, dataset_targets, layer_meta, layer_output)
}
