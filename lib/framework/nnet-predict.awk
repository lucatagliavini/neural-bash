BEGIN {
	# In predict non serve val split (val_split=0)
	load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets,
	              _dummy_vw, _dummy_vt, 0,
	              layer_meta, layer_weights)

	# Applica normalizzazione se il modello è stato addestrato con --normalize
	if (load_norm_stats(model_dir, pred_norm_stats)) {
		apply_normalization(dataset_weights, pred_norm_stats)
		printf("[INFO] predict: input normalization applied (z-score from %s/normalize.conf)\n", model_dir)
	}

	printf("[INFO] predict: Starting inference on %d samples\n", dataset_meta["num_samples"])
	printf("[INFO] predict: Model loaded from: %s\n", model_dir)
	printf("\n")

	# Eseguo il forward pass per ottenere le predizioni (dropout disabilitato):
	forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output, 0, 0, _dm, _lp)

	# Stampo le predizioni con formato dettagliato:
	print_predictions(dataset_meta, dataset_targets, layer_meta, layer_output)

	# Calcolo e stampo metriche finali:
	print_metrics(dataset_meta, dataset_targets, layer_meta, layer_output)
}

