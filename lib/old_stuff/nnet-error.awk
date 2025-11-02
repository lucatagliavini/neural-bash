# awk: lib/functions/nnet-error.awk
#
# Author: Luca Tagliavini
#
# Parametri:
# awk \
#  -v output_file="tmp/xor-test/layer2.out" \
#  -v target_file="tmp/xor-test/target_only.txt" \
#  -v debug=0 \
#  -f lib/functions/nnet-error.awk
#
BEGIN {
    if (debug) print "# Reading output file:", output_file > "/dev/stderr"
    if (debug) print "# Reading target file:", target_file > "/dev/stderr"

    n = 0
    sum_sq_error = 0

    while ((getline out_line < output_file) > 0 && (getline target < target_file) > 0) {
        split(out_line, tokens, /[ \t]+/)
        # Se ci sono più neuroni in output, calcoliamo media per riga
        predicted = 0
        for (i = 1; i <= length(tokens); i++) {
            predicted += tokens[i]
        }
        predicted /= length(tokens)

        error = target - predicted
        sum_sq_error += (error * error)
        n++
    }

    close(output_file)
    close(target_file)

    if (n > 0) {
        mse = sum_sq_error / n
        printf "MSE=%.6f\n", mse
    } else {
        print "ERROR: No data read!" > "/dev/stderr"
        exit 1
    }
}

