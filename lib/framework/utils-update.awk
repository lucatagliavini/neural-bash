#
# File che implementa l'ultimo "step" del training della Neural Netowrk: UPDATE
#

# Funzione di update dei pesi dalla matrice delta:
function update_pass(dataset_meta, dataset_weights, layer_meta, layer_weights, weight_velocity, layer_output, layer_deltas, 
		learning_rate, gradient_clip, optimizer, weight_m, weight_v, adam_beta1, adam_beta2, adam_eps, adam_beta1_t, adam_beta2_t,

		layer_id, layer_id_prev, num_samples, num_layers, num_inputs, neuron, sample, delta, gradient, 
		input_id, input_value, weight_value, prev_v, new_v, m_prev, v_prev, m, vv, m_hat, v_hat,
        	t1_corr, t2_corr, denom) {
	
	# Recupero i dati necessari e loggo:
	num_samples = dataset_meta["num_samples"]
	num_layers = layer_meta[0, 0, 0]

	# Non forziamo upper bound ma se e' > 1.0 diamo messaggio in debug:
	if (momentum > 1.0 && debug_update) {
		logmesg(debug_update, "[WARN] update: momentum   = " momentum " > 1.0, training may be unstable\n")
	}
	logmesg(debug_update, "[DEBUG] update: optimizer     = " optimizer "\n")

	# Se impostato stampiamo il gradient_clip:
	if (gradient_clip != "" && gradient_clip > 0.0) {
		logmesg(debug_update, "[DEBUG] update: gradient_clip = " gradient_clip "\n")
	}

	# Debug:
	logmesg(debug_update, "[DEBUG] update: num_samples   = "num_samples"\n")	
	logmesg(debug_update, "[DEBUG] update: num_layers    = "num_layers"\n")	
	logmesg(debug_update, "[DEBUG] update: learning_rate = "learning_rate"\n")

	# Cicliamo sui layer:
	logmesg(debug_update, "[DEBUG] update: starting cycle on "num_layers" layers\n")
	for (layer_id = 1; layer_id<=num_layers; layer_id++) {
		logmesg(debug_update, "[DEBUG] update: updating layer_id = "layer_id"\n")
	
		# Recuperiamo i dati del layer:
		num_neurons = layer_meta[layer_id, "num_neurons"]
		num_inputs = layer_meta[layer_id, "num_inputs"]

		# Debug:
		logmesg(debug_update, "[DEBUG] update: num_neurons = "num_neurons"\n")
		logmesg(debug_update, "[DEBUG] update: num_inputs = "num_inputs"\n")

		# Aggiornamento neuroni:
		for (neuron = 1; neuron<=num_neurons; neuron++) {

			# Aggiornamento per ogni input:
			for (input_id = 1; input_id<=num_inputs; input_id++) {
				
				# Sommiamo il gradiente su tutti i campioni
				gradient = 0
				for (sample = 1; sample<=num_samples; sample++) {
					
					# Delta calcolo:
					delta = layer_deltas[layer_id, sample, neuron]

					# Troviamo input corretto:
					if (layer_id == 1) {
						# Primo layer: input viene dal dataset (dataset_weights)
						input_value = dataset_weights[sample, input_id]	
					}
					else {
						# Layer successivi: input = output del layer precedente:
						layer_id_prev = layer_id - 1
						input_value = layer_output[layer_id_prev, sample, input_id]
					}

					# Aggiungo al gradiente:
					gradient += delta * input_value
				}
				# Fine ciclo SAMPLE.

				# Calcoliamo la media:
				gradient = gradient / num_samples

				# Implementiamo il GRADIENT-CLIPPING:
				if (gradient_clip != "" && gradient_clip > 0.0) {
					gradient = clip_gradient(debug_update, gradient, gradient_clip)
				}

				# Aggiornamento del peso:
				weight_value = layer_weights[layer_id, neuron, input_id]
					
				# ====================================================================================
				# GESTIONE MOMENTUM e OPTIMIZER
				# ====================================================================================
				if (optimizer == "adam") { # Ramo ADAM:
					# -----------------------------
                    			# Adam optimizer per questo peso
                    			# -----------------------------
                    			# Momento precedente per questo peso
                    			m_prev = weight_m[layer_id, neuron, input_id]
                    			v_prev = weight_v[layer_id, neuron, input_id]

                    			# Aggiorna m e v
                    			m = adam_beta1 * m_prev + (1 - adam_beta1) * gradient
                    			vv = adam_beta2 * v_prev + (1 - adam_beta2) * gradient * gradient

                    			# Salva nuovi momenti
                    			weight_m[layer_id, neuron, input_id] = m
                    			weight_v[layer_id, neuron, input_id] = vv

                    			# Correzione del bias
                    			t1_corr = 1.0 - adam_beta1_t
                    			t2_corr = 1.0 - adam_beta2_t

                    			# Per sicurezza evitiamo divisione per zero
                    			if (t1_corr <= 0) t1_corr = 1e-8
                    			if (t2_corr <= 0) t2_corr = 1e-8

                    			m_hat = m / t1_corr
                    			v_hat = vv / t2_corr

                    			# Denominatore con eps
                    			denom = sqrt(v_hat) + adam_eps
                    			if (denom <= 0) denom = adam_eps

                    			# Update del peso
                    			layer_weights[layer_id, neuron, input_id] = weight_value - (learning_rate * m_hat / denom)

					# Debug apprendimento:
					logmesg(debug_update, "[DEBUG] update(adam-STUB): layer=" layer_id " neuron=" neuron \
					" input=" input_id " beta1=" adam_beta1 " beta2=" adam_beta2 " eps=" adam_eps \
					" weight_old=" weight_value " gradient=" gradient \
					" weight_new=" layer_weights[layer_id, neuron, input_id]"\n")
				}
				else if (momentum > 0.0) { # MOMENTUM POSITIVO (usiamo matrice di velocita')
					# Velocita' precedente (0 se non esiste)
					prev_v = weight_velocity[layer_id, neuron, input_id]

					# Velocirta' calcolata:
					new_v = momentum * prev_v - (learning_rate * gradient)

					# Salviamo la nuova velocita':
					weight_velocity[layer_id, neuron, input_id] = new_v

					# Aggiornamento del peso tenendo conto della velocita':
					layer_weights[layer_id, neuron, input_id] = weight_value + new_v

					# Debug apprendimento con momentum
                    			logmesg(debug_update, "[DEBUG] update(momentum): layer=" layer_id " neuron=" neuron \
                        		" input=" input_id " weight_old=" weight_value " gradient=" gradient \
                        		" prev_vel=" prev_v " new_vel=" new_v " weight_new=" layer_weights[layer_id, neuron, input_id] "\n")
				}
				else { # PLAIN SGD (comportamento senza momentum)
					# Aggiornamento corretto del layer weight, con il "-" anziche' "+":
					layer_weights[layer_id, neuron, input_id] = weight_value - (learning_rate * gradient)

					# Debug apprendimento:
					logmesg(debug_update, "[DEBUG] update: layer=" layer_id " neuron=" neuron \
					" input=" input_id " weight_old=" weight_value " gradient=" gradient \
					" weight_new=" layer_weights[layer_id, neuron, input_id]"\n")
				}
			}  
			# End ciclo INPUTS.
		}
		# End ciclo NEURONI del layer.
	}
	# End ciclo LAYERS.
}
