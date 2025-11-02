#!/usr/bin/awk -f
#
# Script AWK per inizializzare i pesi di una neural network
#
# Usage:
#   awk -f nnet-init.awk \
#       -v model_dir="models/xor" \
#       -v architecture="2,3,1" \
#       -v activation="sigmoid" \
#       -v init_method="xavier" \
#       -v seed=42 \
#       /dev/null
#

BEGIN {
    # Parametri di default
    if (activation == "") activation = "sigmoid"
    if (init_method == "") init_method = "xavier"
    
    # Imposta seed se fornito
    if (seed != "") {
        srand(seed)
        printf("[INFO] Using random seed: %d\n", seed) > "/dev/stderr"
    } else {
        srand()
    }
    
    # Valida parametri
    if (model_dir == "") {
        print "[ERROR] model_dir parameter is required" > "/dev/stderr"
        exit 1
    }
    
    if (architecture == "") {
        print "[ERROR] architecture parameter is required" > "/dev/stderr"
        exit 1
    }
    
    # Valida activation function
    if (activation != "sigmoid" && activation != "tanh" && 
        activation != "relu" && activation != "leaky_relu") {
        printf("[ERROR] Invalid activation: %s\n", activation) > "/dev/stderr"
        printf("Available: sigmoid, tanh, relu, leaky_relu\n") > "/dev/stderr"
        exit 1
    }
    
    # Valida init method
    if (init_method != "xavier" && init_method != "he" && init_method != "random") {
        printf("[ERROR] Invalid init_method: %s\n", init_method) > "/dev/stderr"
        printf("Available: xavier, he, random\n") > "/dev/stderr"
        exit 1
    }
    
    # Parsing architettura
    num_layer_sizes = split(architecture, layer_sizes, ",")
    
    if (num_layer_sizes < 2) {
        print "[ERROR] Architecture must have at least 2 layers" > "/dev/stderr"
        exit 1
    }
    
    # Converti in numeri
    for (i = 1; i <= num_layer_sizes; i++) {
        layer_sizes[i] = layer_sizes[i] + 0
        if (layer_sizes[i] < 1) {
            printf("[ERROR] Layer %d has invalid size: %d\n", i, layer_sizes[i]) > "/dev/stderr"
            exit 1
        }
    }
    
    # Numero di layer (escluso input)
    num_layers = num_layer_sizes - 1
    
    # Crea directory se non esiste
    cmd = "mkdir -p \"" model_dir "\""
    system(cmd)
    
    # Stampa info
    printf("\n") > "/dev/stderr"
    printf("==========================================\n") > "/dev/stderr"
    printf("NEURAL NETWORK INITIALIZATION\n") > "/dev/stderr"
    printf("==========================================\n") > "/dev/stderr"
    printf("Model directory  : %s\n", model_dir) > "/dev/stderr"
    printf("Architecture     : %s\n", architecture) > "/dev/stderr"
    printf("Activation       : %s\n", activation) > "/dev/stderr"
    printf("Init method      : %s\n", init_method) > "/dev/stderr"
    printf("==========================================\n") > "/dev/stderr"
    printf("\n") > "/dev/stderr"
    
    # Crea ogni layer
    total_weights = 0
    
    for (layer_id = 1; layer_id <= num_layers; layer_id++) {
        fan_in = layer_sizes[layer_id]
        fan_out = layer_sizes[layer_id + 1]
        
        num_neurons = fan_out
        num_inputs = fan_in + 1  # include bias
        num_weights = num_neurons * num_inputs
        total_weights += num_weights
        
        layer_file = model_dir "/layer" layer_id ".txt"
        
        printf("[INFO] Creating layer %d: %d neurons, %d inputs (%d weights)\n", 
               layer_id, num_neurons, num_inputs, num_weights) > "/dev/stderr"
        
        # Scrivi header
        printf("ACTIVATION=%s\n", activation) > layer_file
        
        # Genera pesi
        for (neuron = 1; neuron <= num_neurons; neuron++) {
            row = ""
            for (input = 1; input <= num_inputs; input++) {
                weight = generate_weight(init_method, fan_in, fan_out)
                
                if (input < num_inputs) {
                    row = row sprintf("%.6f ", weight)
                } else {
                    row = row sprintf("%.6f", weight)
                }
            }
            printf("%s\n", row) >> layer_file
        }
        
        close(layer_file)
    }
    
    # Summary
    printf("\n") > "/dev/stderr"
    printf("==========================================\n") > "/dev/stderr"
    printf("INITIALIZATION COMPLETED!\n") > "/dev/stderr"
    printf("==========================================\n") > "/dev/stderr"
    printf("\n") > "/dev/stderr"
    printf("Model structure:\n") > "/dev/stderr"
    printf("  Input layer:  %d neurons\n", layer_sizes[1]) > "/dev/stderr"
    
    for (layer_id = 1; layer_id <= num_layers; layer_id++) {
        num_neurons = layer_sizes[layer_id + 1]
        num_inputs = layer_sizes[layer_id]
        num_weights = num_neurons * (num_inputs + 1)
        
        if (layer_id == num_layers) {
            printf("  Output layer: %d neurons (%d weights)\n", 
                   num_neurons, num_weights) > "/dev/stderr"
        } else {
            printf("  Hidden layer %d: %d neurons (%d weights)\n", 
                   layer_id, num_neurons, num_weights) > "/dev/stderr"
        }
    }
    
    printf("\n") > "/dev/stderr"
    printf("Total layers: %d\n", num_layers) > "/dev/stderr"
    printf("Total weights: %d\n", total_weights) > "/dev/stderr"
    printf("\n") > "/dev/stderr"
    printf("Files created in: %s/\n", model_dir) > "/dev/stderr"
    printf("==========================================\n") > "/dev/stderr"
}

# Funzione per generare pesi
function generate_weight(method, fan_in, fan_out,    limit, stddev, u1, u2, z) {
    if (method == "xavier") {
        # Xavier: uniform[-limit, limit] where limit = sqrt(6/(fan_in+fan_out))
        limit = sqrt(6 / (fan_in + fan_out))
        return (rand() * 2 - 1) * limit
        
    } else if (method == "he") {
        # He: normal(0, sqrt(2/fan_in))
        stddev = sqrt(2 / fan_in)
        
        # Box-Muller transform per generare normale
        u1 = rand()
        u2 = rand()
        z = sqrt(-2 * log(u1)) * cos(2 * 3.14159265359 * u2)
        
        return z * stddev
        
    } else if (method == "random") {
        # Random uniform [-0.5, 0.5]
        return rand() - 0.5
        
    } else {
        printf("[ERROR] Unknown method: %s\n", method) > "/dev/stderr"
        exit 1
    }
}
