BEGIN {
	# Simula metadati
	layer_meta[0, 0, 0] = 3
	layer_meta[1, "num_neurons"] = 4
	layer_meta[1, "activation"] = "relu"

	# Test get_num_layers
	n = get_num_layers(layer_meta)
	print "Num layers: ", n

	# Test get Layer Info
	get_layer_info(layer_meta, 1, info)
	print "Layer 1 neurons: ", info["num_neurons"]
	print "Layer 1 activation: ", info["activation"]

	# Test validazioni:
	print "Validate relu: ", validate_activation("relu")
	print "Validate invalid: ", validate_activation("invalid_func")
	
}
