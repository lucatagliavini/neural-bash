BEGIN {
	# Numero di epoche:
	if (max_epochs == "" || max_epochs == 0) {
		max_epochs = 1000
	}	

	# Carico la rete:
	load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets, layer_meta, layer_weights)

	printf("[INFO] train: num_epochs = "max_epochs"\n")
	for (epoch_id=1; epoch_id<=max_epochs; epoch_id++) {
		# Eseguo il forward pass:
		forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output)

		# Eseguo il backward pass:
		backward_pass(dataset_meta, dataset_targets, layer_meta, layer_weights, layer_output, layer_deltas)

		# Eseguo l'update pass:
		update_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, layer_output, layer_deltas, learning_rate)

		# Calcolo l'errore:
		error = compute_mse(dataset_meta, dataset_targets, layer_meta, layer_output)

		# Stampiamo solo se epoch e' ogni 100:
		if (epoch_id == 1 || epoch_id == max_epochs || epoch_id % 100 == 0) {
			printf("[EPOCH %d] MSE = %.6f\n", epoch_id, error)
		}
	}

	# Salvo i pesi aggiornati dopo il training:
	if (save_model == "" || save_model == 1) {
		printf("[INFO] train: saving updated weights to %s\n", model_dir)
		save_nnetwork(model_dir, num_layers, layer_meta, layer_weights)
	}

	# Stampo le predizioni finali se richiesto:
	if (print_result == 1) {
		printf("\n")
		print_final_predictions(dataset_meta, dataset_targets, layer_meta, layer_output)
	}
}

# Funzione per stampare le predizioni finali dopo il training:
function print_final_predictions(dataset_meta, dataset_targets, layer_meta, layer_output,
				num_samples, num_outputs, num_layers, sample, neuron, 
				pred, target, correct, threshold) {
	
	# Estraggo i metadati:
	num_samples = dataset_meta["num_samples"]
	num_outputs = dataset_meta["num_outputs"]
	num_layers = layer_meta[0, 0, 0]
	threshold = 0.5

	printf("============================================================\n")
	printf("FINAL PREDICTIONS\n")
	printf("============================================================\n")

	for (sample = 1; sample <= num_samples; sample++) {
		# Costruisci stringa predizioni:
		pred_str = ""
		for (neuron = 1; neuron <= num_outputs; neuron++) {
			pred = layer_output[num_layers, sample, neuron]
			pred_str = pred_str sprintf("%.6f ", pred)
		}
		
		# Costruisci stringa target:
		target_str = ""
		for (neuron = 1; neuron <= num_outputs; neuron++) {
			target = dataset_targets[sample, neuron]
			target_str = target_str sprintf("%s ", target)
		}

		# Verifica correttezza:
		correct = 1
		for (neuron = 1; neuron <= num_outputs; neuron++) {
			pred = layer_output[num_layers, sample, neuron]
			target = dataset_targets[sample, neuron]
			pred_class = (pred >= threshold ? 1 : 0)
			
			if (pred_class != target) {
				correct = 0
				break
			}
		}
		
		# Stampa riga formattata:
		status = (correct ? "✓" : "✗")
		printf("[Sample %d] pred = %-20s | target = %-10s | %s\n", 
		       sample, pred_str, target_str, status)
	}
	printf("============================================================\n")
}
