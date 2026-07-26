#
# Funzioni per caricare in memoria i metadati di una rete neurale
# ad N layer.
#
# Servono le seguenti matrici:
#
# - layer_meta
# - layer_weights
# - layer_output
# - layer_delta
# - layer_gradients

# Carichiamo in memoria la neural network:
#
# OUTPUT:
# - Matrice: layer_meta
#   layer_meta[id, "n_neurons"] = 2
#   layer_meta[id, "n_inputs"]  = 3
#   layer_meta[id, "activation"] = sigmoid
#
# - Matrice: 
#   layer_weights[id, 1, 2] = layer (id), neurone 1, input 2
#
# 

# Carica la rete neurale:
function load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets,
			val_weights, val_targets, val_split,
			layer_meta, layer_weights,
			layer_id) {
	logmesg(debug_network, "[DEBUG] nnetwork: dataset_file = " dataset_file"\n")
	logmesg(debug_network, "[DEBUG] nnetwork: num_inputs = " num_inputs"\n")

	# Caricamento dataset con eventuale val split
	load_dataset(dataset_file, num_inputs, dataset_meta, dataset_weights, dataset_targets,
	             val_weights, val_targets, val_split)
	
	# Stampa solo se abbiamo valorizzato il flag.
	logmesg(debug_network, "[DEBUG] nnetwork: dataset_meta matrix:\n")
	logmesg(debug_network, "[DEBUG] nnetwork: dataset_meta[num_samples] = "dataset_meta["num_samples"]"\n")
	logmesg(debug_network, "[DEBUG] nnetwork: dataset_meta[num_inputs] = "dataset_meta["num_inputs"]"\n")
	logmesg(debug_network, "[DEBUG] nnetwork: dataset_meta[num_outputs] = "dataset_meta["num_outputs"]"\n")

	logmesg(debug_network, "[DEBUG] nnetwork: dataset_weigths matrix:\n")
	logmatrix(debug_network, dataset_weights)

	logmesg(debug_network, "[DEBUG] nnewtork: dataset_targets matrix:\n")
	logmatrix(debug_network, dataset_targets)

	# Caricamento del model (tutti i layers)
	logmesg(debug_network, "[DEBUG] nnetwork: model_dir = "model_dir"\n")
	logmesg(debug_network, "[DEBUG] nnetwork: num_layers = "num_layers"\n")

	# Carico i layers:
	load_layers(model_dir, num_layers, layer_meta, layer_weights)
	
	# Stampiamo debug:
	for (layer_id = 1; layer_id<=num_layers; layer_id++) {
		# Stampiamo i metadati di ogni layer:
		logmesg(debug_network, "[DEBUG] nnetwork: layer_meta["layer_id", \"activation\"] = "layer_meta[layer_id, "activation"]"\n")
		logmesg(debug_network, "[DEBUG] nnetwork: layer_meta["layer_id", \"num_inputs\"] = "layer_meta[layer_id, "num_inputs"]"\n")
		logmesg(debug_network, "[DEBUG] nnetwork: layer_meta["layer_id", \"num_neurons\"] = "layer_meta[layer_id, "num_neurons"]"\n")
		logmesg(debug_network, "[DEBUG] nnetwork: layer_meta["layer_id", \"has_bias\"] = "layer_meta[layer_id, "has_bias"]"\n")
		
		# Stampiamo i pesi caricati:
		logmesg(debug_network, "[DEBUG] nnetwork: layer_weights matrix:")
		logmatrix_weights(debug_network, layer_weights, layer_id)	
	}
}

# Salva la rete neurale dopo il training:
function save_nnetwork(model_dir, num_layers, layer_meta, layer_weights,
			layer_id, layer_file, num_neurons, num_inputs, row, col, activation_function, activation_str) {
	
	# Verifica che la directory esista:
	if (system("[ -d \"" model_dir "\" ]") != 0) {
		printf("[ERROR] save_nnetwork: directory not found: %s\n", model_dir) > "/dev/stderr"
		return 0
	}

	# Salvo ogni layer:
	for (layer_id = 1; layer_id <= num_layers; layer_id++) {
		layer_file = model_dir "/layer" layer_id ".txt"
		
		# Estraggo metadati del layer:
		activation_function = layer_meta[layer_id, "activation"]
		num_neurons = layer_meta[layer_id, "num_neurons"]
		num_inputs = layer_meta[layer_id, "num_inputs"]

		# Ricompongo "leaky_relu:alpha" se necessario, altrimenti uso solo il nome
		if (activation_function == "leaky_relu") {
			activation_str = activation_function ":" layer_meta[layer_id, "alpha"]
		} else {
			activation_str = activation_function
		}

		# Scrivo il file (sovrascrivo quello esistente):
		printf("ACTIVATION=%s\n", activation_str) > layer_file
		
		# Scrivo i pesi:
		for (row = 1; row <= num_neurons; row++) {
			for (col = 1; col <= num_inputs; col++) {
				if (col < num_inputs) {
					printf("%.10g ", layer_weights[layer_id, row, col]) >> layer_file
				} else {
					printf("%.10g", layer_weights[layer_id, row, col]) >> layer_file
				}
			}
			printf("\n") >> layer_file
		}
		
		close(layer_file)
		logmesg(debug_network, "[DEBUG] save_nnetwork: saved layer " layer_id " to " layer_file "\n")
	}
	
	return 1
}
