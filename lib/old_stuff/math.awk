# lib/math.awk

# ============================================
# Funzioni matematiche base per reti neurali
# ============================================

# Converte una riga in array numerico
function parse_line_to_array(line, arr,    i, n, val) {
    n = split(line, arr, /[ \t]+/)
    for (i = 1; i <= n; i++) arr[i] += 0
    return n
}

# Prodotto scalare tra due vettori
function dot_product(a, b, len,    sum, i) {
    sum = 0
    for (i = 1; i <= len; i++) sum += a[i] * b[i]
    return sum
}

# Somma di due vettori
function add_vectors(a, b, result, len,    i) {
    for (i = 1; i <= len; i++) result[i] = a[i] + b[i]
}

# Sottrazione di due vettori
function sub_vectors(a, b, result, len,    i) {
    for (i = 1; i <= len; i++) result[i] = a[i] - b[i]
}

# Moltiplica un vettore per uno scalare
function scale_vector(v, scalar, result, len,    i) {
    for (i = 1; i <= len; i++) result[i] = v[i] * scalar
}

# Moltiplicazione matrice * vettore
# La matrice deve essere un array bidimensionale: matrix[row,col]
function matrix_vector_mul(matrix, rows, cols, vector, result,    r, c, sum) {
    for (r = 1; r <= rows; r++) {
        sum = 0
        for (c = 1; c <= cols; c++) {
            sum += matrix[r, c] * vector[c]
        }
        result[r] = sum
    }
}

# Moltiplicazione vettore * matrice (vettore riga × matrice colonnare)
# Per step backward.
function vector_matrix_mul(vector, matrix, rows, cols, result,    i, j, sum) {
    for (j = 1; j <= cols; j++) {
        sum = 0
        for (i = 1; i <= rows; i++) {
            sum += vector[i] * matrix[i, j]
        }
        result[j] = sum
    }
}

# Trasposizione di matrice: output[c,r] = input[r,c]
function transpose_matrix(input, rows, cols, output,    r, c) {
    for (r = 1; r <= rows; r++)
        for (c = 1; c <= cols; c++)
            output[c, r] = input[r, c]
}

# Parsing di layer file in matrice (salta la riga di attivazione e i commenti)
function load_weight_matrix(filename, matrix, nrows, ncols,    line, row, tmp) {
    row = 0
    while ((getline line < filename) > 0) {
	if (line ~ /^ACTIVATION=/ || line ~ /^#/ || line ~ /^ROWS=/ || line ~ /^COLS=/) continue
        row++
        ncols = parse_line_to_array(line, tmp)
        for (i = 1; i <= ncols; i++) matrix[row, i] = tmp[i]
    }
    close(filename)
    nrows = row
    return nrows
}

# Stampa un vettore su stdout
function print_vector(v, len, out_stream,    i) {
    for (i = 1; i <= len; i++) printf("%s%s", v[i], (i<len ? " " : "\n")) > out_stream
}

# Stampa una matrice
function print_matrix(matrix, rows, cols, out_stream,    r, c) {
    for (r = 1; r <= rows; r++) {
        for (c = 1; c <= cols; c++) {
            printf("%s%s", matrix[r,c], (c<cols ? " " : "\n")) > out_stream
        }
    }
}
