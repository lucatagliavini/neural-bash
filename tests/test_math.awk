# tests/test_math.awk

BEGIN {
    # Vettore input fittizio (dimensione = colonne del layer)
    input[1] = 1.0
    input[2] = 2.0
    input[3] = 3.0
    input[4] = 1.0  # Bias implicito

    # Caricamento matrice pesi (3 neuroni × 4 input)
    layer_file = "layer_matrix.txt"
    rows = load_weight_matrix(layer_file, W, nrows, ncols)

    print "[INFO] Matrice pesi caricata:"
    print_matrix(W, nrows, ncols)

    print "[INFO] Vettore input:"
    print_vector(input, ncols)

    # Moltiplicazione matrice × vettore
    matrix_vector_mul(W, nrows, ncols, input, output)

    print "[INFO] Output della rete (prima dell'attivazione):"
    print_vector(output, nrows)

    exit
}
