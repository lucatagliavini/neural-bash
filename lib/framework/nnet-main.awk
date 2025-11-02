BEGIN {

	# Carico la rete:
	load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets, layer_meta, layer_weights)

	# Eseguo il forward pass:
	forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output)

	# Eseguo il backward pass:
	backward_pass(dataset_meta, dataset_targets, layer_meta, layer_weights, layer_output, layer_deltas)

	# Eseguo l'update pass:
	update_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, layer_output, layer_deltas, learning_rate)

	# Calcolo l'errore:
	epoch_id=1
	error = compute_mse(dataset_meta, dataset_targets, layer_meta, layer_output)
	printf("[EPOCH %d] MSE = %.6f\n", epoch_id, error)
}
