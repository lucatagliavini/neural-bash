BEGIN {
    # Preparo dati
    activation = "sigmoid"
    nrows = 2
    ncols = 3
    weights[1,1] = 0.5
    weights[1,2] = 0.3
    weights[1,3] = -0.1
    weights[2,1] = 0.2
    weights[2,2] = -0.4
    weights[2,3] = 0.7

    # Salvo su file
    write_weights("test-layer.txt", weights, activation, nrows, ncols)
}
