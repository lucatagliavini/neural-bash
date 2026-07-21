BEGIN {
	# Carico la rete neurale:
	load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets, layer_meta, layer_weights)

	printf("[INFO] predict: Starting inference on %d samples\n", dataset_meta["num_samples"])
	printf("[INFO] predict: Model loaded from: %s\n", model_dir)
	printf("\n")

	# Eseguo il forward pass per ottenere le predizioni:
	forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output)

	# Stampo le predizioni con formato dettagliato:
	print_predictions(dataset_meta, dataset_targets, layer_meta, layer_output)

	# Calcolo e stampo metriche finali:
	print_metrics(dataset_meta, dataset_targets, layer_meta, layer_output)
}

