BEGIN {
	# Numero di epoche:
	if (max_epochs == "" || max_epochs == 0) {
		max_epochs = 1000
	}

	# Settiamo la learning rate:
	if (learning_rate == "" || learning_rate == 0) {
		learning_rate = 0.1
	}

	# Gestiamo il momentum:
	if (momentum == "" || momentum < 0.0) {
		momentum = 0.0
	}

	# Gestiamo il gradient_clipping:
	if (gradient_clip == "" || gradient_clip < 0.0) {
		gradient_clip = 0.0
	}

	# Gestiamo optimizer:
	if (optimizer == "") {
		optimizer="sgd"
	}
	# Parametri Adam (default standard)
	if (optimizer == "adam") {
    		adam_beta1 = 0.9
    		adam_beta2 = 0.999
    		adam_eps   = 1e-8
	} 
	else {
    		adam_beta1 = 0.0
    		adam_beta2 = 0.0
    		adam_eps   = 0.0
	}
	# Potenze per correzione del bias (beta1^t, beta2^t)
	adam_beta1_t = 1.0
	adam_beta2_t = 1.0

	# Normalizzazione del Learning Rate:
	if (lr_decay == "" || lr_decay < 0.0) {
		lr_decay = 0.0
	}
	# Settiamo la base di partenza e l'attuale:
	base_learning_rate = learning_rate
	current_lr = base_learning_rate

	# Logging parametri:
	logmesg(debug_network, 	"[INFO] train: optimizer=" optimizer \
							", base_lr=" base_learning_rate ", lr_decay=" + lr_decay ", momentum=" momentum "\n")

	# Settato anche:
	if (gradient_clip != "") {
		logmesg(debug_network, "[INFO] train: gradient_clip = " gradient_clip "\n")
	}

	# Carico la rete:
	load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets, layer_meta, layer_weights)
	
	printf("[INFO] train: num_epochs = "max_epochs"\n")
	for (epoch_id=1; epoch_id<=max_epochs; epoch_id++) {
		# Se abbiamo decay di LR: (altrimenti rimane costante)
		if (lr_decay > 0.0) {
			current_lr = base_learning_rate / (1 + lr_decay * (epoch_id - 1))
		}

		# Aggiornamento potenze di beta1/beta2 per Adam (per correzione bias)
    		if (optimizer == "adam") {
        		adam_beta1_t *= adam_beta1
        		adam_beta2_t *= adam_beta2
    		}

		# Eseguo il forward pass:
		forward_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, layer_output, layer_preactivation)

		# Eseguo il backward pass:
		backward_pass(dataset_meta, dataset_targets, layer_meta, layer_weights, layer_output, layer_preactivation, layer_deltas)

		# Eseguo l'update pass:
		update_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, weight_velocity, layer_output, layer_deltas, current_lr, 
					gradient_clip, optimizer, weight_m, weight_v, adam_beta1, adam_beta2, adam_eps, 
					adam_beta1_t, adam_beta2_t)

		# Calcolo l'errore [mse e loss function]:
		mse = compute_mse(dataset_meta, dataset_targets, layer_meta, layer_output)
		loss = compute_dataset_loss(dataset_meta, dataset_targets, layer_meta, layer_output, loss_function)

		# Stampiamo solo se epoch e' ogni 100:
		if (epoch_id == 1 || epoch_id == max_epochs || epoch_id % 100 == 0) {
			printf("[EPOCH %d] MSE = %.6f | LR = %.6f | LOSS(%s) = %.6f\n", epoch_id, mse, current_lr, loss_function, loss)
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
