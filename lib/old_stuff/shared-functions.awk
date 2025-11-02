##############################################################################
# shared-functions.awk – Funzioni comuni a tutti i moduli nnet-*.awk
# Progetto: Neural Net Bash+AWK Framework
# Autore: [Tuo Nome o Luca Tagliavini]
# Versione: iniziale con supporto pesi/input/gradienti
##############################################################################

# Legge un file layer con intestazione e ritorna:
# - activation = nome funzione di attivazione (es. sigmoid)
# - nrows = numero di righe (neuroni)
# - ncols = numero di colonne (input + bias)
# - weights[i,j] = peso del neurone i sulla colonna j
function read_weights(file, weights, activation, nrows, ncols,   line, i, j, row, cols, fields) {
    g_nrows = 0
    g_ncols = 0
    g_activation = ""
    row = 0

    while ((getline line < file) > 0) {
        if (line ~ /^ACTIVATION=/) {
            sub(/^ACTIVATION=/, "", line)
            g_activation = line
            if (debug) print "# Found ACTIVATION =", g_activation > "/dev/stderr"
            continue
        }
        if (line ~ /^ROWS=/) {
            sub(/^ROWS=/, "", line)
            g_nrows = line + 0
            if (debug) print "# Found ROWS =", g_nrows > "/dev/stderr"
            continue
        }
        if (line ~ /^COLS=/) {
            sub(/^COLS=/, "", line)
            g_ncols = line + 0
            if (debug) print "# Found COLS =", g_ncols > "/dev/stderr"
            continue
        }

        # Riga di pesi: es. "0.3 0.4 -0.1"
        ++row
        cols = split(line, fields, /[ \t]+/)
	if (debug) print "# Row", row, "has", cols, "columns" > "/dev/stderr"
        for (j = 1; j <= cols; j++) {
            weights[row, j] = fields[j]
	    if (debug) print "# weights[" row "," j "] = " weights[row, j] > "/dev/stderr"
        }
    }

    close(file)
    if (debug) print "# RETURNING: activation=" g_activation ", rows=" g_nrows ", cols=" g_ncols > "/dev/stderr"
    return (g_activation != "" && g_nrows > 0 && g_ncols > 0)
}

# Scrive una matrice di pesi su file in formato layer
# - file = destinazione
# - weights[i,j] = matrice di pesi
# - activation = funzione di attivazione (es. sigmoid)
# - nrows, ncols = dimensioni

function write_weights(file, weights, activation, nrows, ncols, i, j, line) {
    print "ACTIVATION=" activation > file
    print "ROWS=" nrows >> file
    print "COLS=" ncols >> file

    for (i = 1; i <= nrows; i++) {
        line = ""
        for (j = 1; j <= ncols; j++) {
            line = line weights[i, j] " "
        }
        sub(/[ \t]+$/, "", line)
        print line >> file
    }
    close(file)
}

# Legge una matrice da file e la carica in arr[row,col]
# Ritorna numero di righe
function read_matrix(file, arr,   r, line, tmp, i, ncols) {
    r = 0
    while ((getline line < file) > 0) {
        if (line ~ /^ACTIVATION=/ || line ~ /^ROWS=/ || line ~ /^COLS=/) continue
        ++r
        ncols = split(line, tmp, /[ \t]+/)
        for (i = 1; i <= ncols; i++) {
            arr[r, i] = tmp[i]
        }
    }
    close(file)
    return r
}

function write_matrix(file, matrix, nrows, ncols,  row, col, line) {
    for (row = 1; row <= nrows; row++) {
        line = ""
        for (col = 1; col <= ncols; col++) {
            line = line matrix[row, col] " "
        }
        sub(/[ \t]+$/, "", line)
        print line >> file
    }
    close(file)
}

# Legge la matrice input X[k,i] da un file
function read_inputs(file,    i, tokens, line) {
    if (debug) print "# Reading inputs from", file > "/dev/stderr"
    input_rows = 0
    while ((getline line < file) > 0) {
        split(line, tokens, /[ \t]+/)
        for (i = 1; i <= length(tokens); i++) {
            X[input_rows + 1, i] = tokens[i]
        }
        input_rows++
    }
    close(file)
    if (debug) print "# Input examples:", input_rows > "/dev/stderr"
}

# Legge la matrice dei gradienti D[k,j] da un file
function read_gradients(file,    j, tokens, line) {
    if (debug) print "# Reading gradients from", file > "/dev/stderr"
    grad_rows = 0
    while ((getline line < file) > 0) {
        split(line, tokens, /[ \t]+/)
        for (j = 1; j <= length(tokens); j++) {
            D[grad_rows + 1, j] = tokens[j]
        }
        grad_rows++
    }
    close(file)
    if (debug) print "# Gradient rows:", grad_rows > "/dev/stderr"
}

