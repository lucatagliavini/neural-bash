BEGIN {
	# Seed per riproducibilità: se passato usa srand(seed), altrimenti srand() con time
	if (seed != "" && seed > 0) {
		srand(seed)
	} else {
		srand()
	}

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

	# Mini-batch: batch_size=0 o >= num_samples → full-batch
	if (batch_size == "" || batch_size <= 0) {
		batch_size = 0
	}

	# Logging parametri:
	logmesg(debug_network, 	"[INFO] train: optimizer=" optimizer \
							", base_lr=" base_learning_rate ", lr_decay=" + lr_decay ", momentum=" momentum "\n")

	# Carico la rete (val_split=0 → nessuno split, tutti i campioni in train)
	load_nnetwork(dataset_file, num_inputs, model_dir, num_layers, dataset_meta, dataset_weights, dataset_targets,
	              val_weights, val_targets, val_split,
	              layer_meta, layer_weights)

	# Normalizzazione input z-score (opzionale, --normalize)
	if (normalize == 1) {
		compute_norm_stats(dataset_weights, num_inputs, norm_stats)
		apply_normalization(dataset_weights, norm_stats)
		if (dataset_meta["num_val_samples"] > 0)
			apply_normalization(val_weights, norm_stats)
		save_norm_stats(model_dir, norm_stats)
		printf("[INFO] train: input normalization enabled (z-score, saved to %s/normalize.conf)\n", model_dir)
	}

	# Inizializzo il best checkpoint:
	best_mse = 1e9
	best_checkpoint_dir = model_dir "/best"
	system("mkdir -p \"" best_checkpoint_dir "\"")

	# Early stopping: monitora val_mse, si attiva solo se val_split > 0
	if (patience > 0 && (val_split == "" || val_split <= 0)) {
		printf("[WARNING] train: --patience richiede --val-split > 0; early stopping disabilitato\n") > "/dev/stderr"
		patience = 0
	}
	best_val_mse    = 1e9
	patience_count  = 0

	# Determina la modalità e il batch size effettivo
	n_train = dataset_meta["num_samples"]
	use_minibatch = (batch_size > 0 && batch_size < n_train)
	eff_batch = use_minibatch ? batch_size : n_train

	if (use_minibatch) {
		printf("[INFO] train: mini-batch SGD enabled (batch_size=%d, samples=%d)\n", eff_batch, n_train)
	}

	printf("[INFO] train: num_epochs = "max_epochs"\n")
	for (epoch_id=1; epoch_id<=max_epochs; epoch_id++) {
		# Se abbiamo decay di LR: (altrimenti rimane costante)
		if (lr_decay > 0.0) {
			current_lr = base_learning_rate / (1 + lr_decay * (epoch_id - 1))
		}

		if (use_minibatch) {
			# Shuffle Fisher-Yates degli indici 1..n_train
			for (_i = 1; _i <= n_train; _i++) _perm[_i] = _i
			for (_i = n_train; _i >= 2; _i--) {
				_j = int(rand() * _i) + 1
				_tmp = _perm[_i]; _perm[_i] = _perm[_j]; _perm[_j] = _tmp
			}

			# Loop sui mini-batch
			for (_b_start = 1; _b_start <= n_train; _b_start += eff_batch) {
				_b_end = _b_start + eff_batch - 1
				if (_b_end > n_train) _b_end = n_train
				_b_size = _b_end - _b_start + 1

				# Copia campioni del batch in batch_weights/batch_targets
				clear_array(batch_weights)
				clear_array(batch_targets)
				for (_bi = 1; _bi <= _b_size; _bi++) {
					_src = _perm[_b_start + _bi - 1]
					_ncols = dataset_weights[_src, 0]
					for (_c = 0; _c <= _ncols; _c++)
						batch_weights[_bi, _c] = dataset_weights[_src, _c]
					_nout = dataset_meta["num_outputs"]
					for (_c = 1; _c <= _nout; _c++)
						batch_targets[_bi, _c] = dataset_targets[_src, _c]
				}
				batch_meta["num_samples"] = _b_size
				batch_meta["num_inputs"]  = dataset_meta["num_inputs"]
				batch_meta["num_outputs"] = dataset_meta["num_outputs"]

				# Adam: avanza il contatore t ad ogni batch step
				if (optimizer == "adam") {
					adam_beta1_t *= adam_beta1
					adam_beta2_t *= adam_beta2
				}

				clear_array(dropout_mask)
				clear_array(layer_preact)
				forward_pass(batch_meta, batch_weights, num_layers, layer_meta, layer_weights, layer_output,
				             (dropout > 0), dropout, dropout_mask, layer_preact)
				backward_pass(batch_meta, batch_targets, layer_meta, layer_weights, layer_output, layer_deltas, loss_function,
				              dropout_mask, layer_preact)
				update_pass(batch_meta, batch_weights, layer_meta, layer_weights, weight_velocity, layer_output, layer_deltas, current_lr,
				            optimizer, weight_m, weight_v, adam_beta1, adam_beta2, adam_eps, adam_beta1_t, adam_beta2_t,
				            grad_clip, weight_decay)
			}

			# Forward completo sull'intero training set per calcolare MSE/loss dell'epoca
			forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output, 0, 0, _dm, _lp)

		} else {
			# Full-batch: comportamento originale
			# Aggiornamento potenze di beta1/beta2 per Adam
			if (optimizer == "adam") {
				adam_beta1_t *= adam_beta1
				adam_beta2_t *= adam_beta2
			}

			clear_array(dropout_mask)
			clear_array(layer_preact)
			forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output,
			             (dropout > 0), dropout, dropout_mask, layer_preact)
			backward_pass(dataset_meta, dataset_targets, layer_meta, layer_weights, layer_output, layer_deltas, loss_function,
			              dropout_mask, layer_preact)
			update_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, weight_velocity, layer_output, layer_deltas, current_lr,
						optimizer, weight_m, weight_v, adam_beta1, adam_beta2, adam_eps, adam_beta1_t, adam_beta2_t,
						grad_clip, weight_decay)
		}

		# Calcolo l'errore [mse e loss function] sull'intero training set:
		mse = compute_mse(dataset_meta, dataset_targets, layer_meta, layer_output)
		loss = compute_dataset_loss(dataset_meta, dataset_targets, layer_meta, layer_output, loss_function)

		# Calcolo val_mse se abbiamo un set di validazione
		val_mse = ""
		if (dataset_meta["num_val_samples"] > 0) {
			val_meta["num_samples"] = dataset_meta["num_val_samples"]
			val_meta["num_inputs"]  = dataset_meta["num_inputs"]
			val_meta["num_outputs"] = dataset_meta["num_outputs"]
			forward_pass(val_meta, val_weights, num_layers, layer_meta, layer_weights, val_output, 0, 0, _dm, _lp)
			val_mse = compute_mse(val_meta, val_targets, layer_meta, val_output)
		}

		# Salvo il best checkpoint se MSE migliora:
		if (mse < best_mse) {
			best_mse = mse
			best_epoch = epoch_id
			save_nnetwork(best_checkpoint_dir, num_layers, layer_meta, layer_weights)
		}

		# Early stopping su val_mse
		if (patience > 0 && val_mse != "") {
			if (val_mse < best_val_mse) {
				best_val_mse   = val_mse
				patience_count = 0
			} else {
				patience_count++
				if (patience_count >= patience) {
					printf("[INFO] train: early stopping at epoch %d (val_mse=%.6f, no improvement for %d epochs)\n",
					       epoch_id, val_mse, patience)
					break
				}
			}
		}

		# Stampiamo solo se epoch e' ogni 100:
		if (epoch_id == 1 || epoch_id == max_epochs || epoch_id % 100 == 0) {
			if (val_mse != "") {
				printf("[EPOCH %d] MSE = %.6f | VAL_MSE = %.6f | LR = %.6f | LOSS(%s) = %.6f\n",
				       epoch_id, mse, val_mse, current_lr, loss_function, loss)
			} else {
				printf("[EPOCH %d] MSE = %.6f | LR = %.6f | LOSS(%s) = %.6f\n", epoch_id, mse, current_lr, loss_function, loss)
			}
		}
	}

	# Salvo i pesi aggiornati dopo il training:
	if (save_model == "" || save_model == 1) {
		printf("[INFO] train: saving updated weights to %s\n", model_dir)
		save_nnetwork(model_dir, num_layers, layer_meta, layer_weights)
	}

	# Scrivi sempre i metadati di sessione (usati da nnet-run.sh per model.conf
	# e da nnet-search.sh anche con --no-save)
	meta_file = model_dir "/.train_meta"
	printf("last_optimizer=%s\n", optimizer)                     > meta_file
	printf("last_lr=%.8g\n",      base_learning_rate)           >> meta_file
	printf("last_lr_decay=%.8g\n", lr_decay)                    >> meta_file
	printf("last_momentum=%.8g\n", momentum)                    >> meta_file
	printf("last_epochs=%d\n",     max_epochs)                  >> meta_file
	printf("last_loss=%s\n",       loss_function)               >> meta_file
	printf("best_mse=%.8g\n",      best_mse)                    >> meta_file
	printf("best_epoch=%d\n",      best_epoch)                  >> meta_file
	close(meta_file)

	printf("[INFO] train: best MSE = %.6f (epoch %d) → saved in %s\n", best_mse, best_epoch, best_checkpoint_dir)

	# Stampo le predizioni finali se richiesto:
	if (print_result == 1) {
		# Riesegue il forward senza dropout per le metriche finali
		forward_pass(dataset_meta, dataset_weights, num_layers, layer_meta, layer_weights, layer_output, 0, 0, _dm, _lp)
		printf("\n")
		print_predictions(dataset_meta, dataset_targets, layer_meta, layer_output)
		print_metrics(dataset_meta, dataset_targets, layer_meta, layer_output)
	}
}
