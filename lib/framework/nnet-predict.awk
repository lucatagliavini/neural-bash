BEGIN {
	# Carico la rete neurale:
	load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets, layer_meta, layer_weights)

	printf("[INFO] predict: Starting inference on %d samples\n", dataset_meta["num_samples"])
	printf("[INFO] predict: Model loaded from: %s\n", model_dir)
	printf("\n")

	# Eseguo il forward pass per ottenere le predizioni:
	forward_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, layer_output, layer_preactivation)

	# Stampo le predizioni con formato dettagliato:
	print_predictions(dataset_meta, dataset_targets, layer_meta, layer_output)

	# Calcolo e stampo metriche finali:
	print_metrics(dataset_meta, dataset_targets, layer_meta, layer_output)
}

# Funzione per stampare le predizioni in formato leggibile:
function print_predictions(dataset_meta, dataset_targets, layer_meta, layer_output,
				num_samples, num_outputs, num_layers, sample, neuron, 
				pred, target, correct, threshold) {
	
	# Estraggo i metadati:
	num_samples = dataset_meta["num_samples"]
	num_outputs = dataset_meta["num_outputs"]
	num_layers = layer_meta[0, 0, 0]
	
	# Threshold per classificazione binaria:
	threshold = 0.5

	printf("================================================================================\n")
	printf("PREDICTIONS\n")
	printf("================================================================================\n")
	printf("%-8s | %-15s | %-10s | %-10s\n", "Sample", "Predicted", "Target", "Status")
	printf("--------------------------------------------------------------------------------\n")

	for (sample = 1; sample <= num_samples; sample++) {
		# Costruisci stringa delle predizioni:
		pred_str = ""
		for (neuron = 1; neuron <= num_outputs; neuron++) {
			pred = layer_output[num_layers, sample, neuron]
			if (neuron == 1) pred_str = sprintf("%.6f", pred)
			else pred_str = pred_str sprintf(" %.6f", pred)
		}
		
		# Costruisci stringa dei target:
		target_str = ""
		for (neuron = 1; neuron <= num_outputs; neuron++) {
			target = dataset_targets[sample, neuron]
			if (neuron == 1) target_str = sprintf("%s", target)
			else target_str = target_str sprintf(" %s", target)
		}

		# Verifica correttezza (per classificazione binaria):
		correct = 1
		for (neuron = 1; neuron <= num_outputs; neuron++) {
			pred = layer_output[num_layers, sample, neuron]
			target = dataset_targets[sample, neuron]
			
			# Classificazione binaria con threshold:
			pred_class = (pred >= threshold ? 1 : 0)
			if (pred_class != target) {
				correct = 0
				break
			}
		}
		
		# Stampa riga completa con allineamento corretto:
		status_str = (correct ? "✓ CORRECT" : "✗ WRONG")
		printf("%-8d | %-15s | %-10s | %-10s\n", sample, pred_str, target_str, status_str)
	}
	printf("================================================================================\n")
	printf("\n")
}

# Funzione per stampare metriche di valutazione:
function print_metrics(dataset_meta, dataset_targets, layer_meta, layer_output,
			num_samples, num_outputs, num_layers, mse, accuracy, 
			sample, neuron, pred, target, correct, total_correct, threshold) {
	
	# Estraggo i metadati:
	num_samples = dataset_meta["num_samples"]
	num_outputs = dataset_meta["num_outputs"]
	num_layers = layer_meta[0, 0, 0]
	threshold = 0.5
	
	# Calcolo MSE:
	mse = compute_mse(dataset_meta, dataset_targets, layer_meta, layer_output)
	
	# Calcolo accuracy per classificazione:
	total_correct = 0
	for (sample = 1; sample <= num_samples; sample++) {
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
		if (correct) total_correct++
	}
	
	accuracy = (total_correct / num_samples) * 100
	
	# Stampa metriche:
	printf("EVALUATION METRICS\n")
	printf("================================================================================\n")
	printf("Mean Squared Error (MSE) : %.6f\n", mse)
	printf("Accuracy                  : %.2f%% (%d/%d)\n", accuracy, total_correct, num_samples)
	printf("================================================================================\n")
}
