# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Approach
- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct. No over-engineering.
- If unsure: say so. Never guess or invent file paths.
- User instructions always override this file.

## Efficiency
- Read before writing. Understand the problem before coding.
- No redundant file reads. Read each file once.
- One focused coding pass. Avoid write-delete-rewrite cycles.
- Test once, fix if needed, verify once. No unnecessary iterations.
- Budget: 50 tool calls maximum. Work efficiently.

## Project Overview

A neural network framework implemented entirely in AWK with Bash wrappers. Supports training (backpropagation), inference, and model persistence. No compilation needed — AWK is interpreted.

## Common Commands

```bash
# Initialize a new network
./nnet-init.sh models/<name> <topology> --activation sigmoid --method xavier
# e.g.: ./nnet-init.sh models/custom 2,3,1

# Train a model
./nnet-run.sh train dataset/xor.txt models/xor --epochs 1000 --lr 0.3

# Run inference
./nnet-run.sh predict dataset/xor.txt models/xor

# Train then evaluate
./nnet-run.sh eval dataset/xor.txt models/xor --epochs 2000

# Run tests
bash tests/run.sh

# Demo
bash demo.sh
```

**Key CLI flags:** `--lr`, `--epochs`, `--optimizer` (sgd|sgd-momentum|sgd-momentum-decay|adam), `--loss` (mse|ce), `--momentum`, `--lr-decay`, `--debug` (forward|backward|update|network|metrics|all), `--no-save`

## Architecture

All ML logic lives in `lib/framework/` as AWK modules loaded via `-f` flags. The Bash scripts (`nnet-run.sh`, `nnet-train.sh`, `nnet-predict.sh`, `nnet-init.sh`) are thin wrappers that parse CLI args and invoke `gawk` with the appropriate modules.

### Training Pipeline

1. **Init** (`nnet-init.awk`) — creates one text file per layer under `models/<name>/layer1.txt`, `layer2.txt`, etc. Each file starts with `ACTIVATION=<fn>` then one row of weights per neuron (last weight is bias).
2. **Forward** (`utils-forward.awk`) — computes `z = W·x`, applies activation, stores results in `layer_output[layer_id, sample, neuron]`.
3. **Backward** (`utils-backward.awk`) — computes gradients into `layer_deltas[layer_id, sample, neuron]`.
4. **Update** (`utils-update.awk`) — applies SGD / Momentum / Adam to `layer_weights`.
5. **Save** (`utils-network.awk`) — writes updated weights back to layer files.

### Key AWK Array Conventions

- `matrix[0,0]` stores the row count; `matrix[i,0]` stores the column count for row `i`.
- Bias is automatically appended as the last input column (always `1.0`); layer weight files include the bias weight as the last element per row.
- `layer_meta[layer_id, "activation"]`, `layer_meta[layer_id, "num_neurons"]`, `layer_meta[layer_id, "num_inputs"]` hold layer configuration.

### Module Responsibilities

| File | Responsibility |
|------|---------------|
| `utils-shared.awk` | Logging (`logmesg`), dataset loading, matrix I/O helpers |
| `utils-network.awk` | `load_nnetwork()` / `save_nnetwork()` — model file I/O |
| `utils-activation.awk` | `apply_activation()` / `apply_activation_derivative()` — sigmoid, tanh, relu, leaky_relu |
| `utils-loss.awk` | `compute_mse()`, `compute_dataset_loss()` — MSE and cross-entropy |
| `utils-math.awk` | Weight initialization (Xavier, He, uniform), Gaussian random |
| `utils-forward.awk` | Full forward pass over all samples and layers |
| `utils-backward.awk` | Backpropagation — delta computation for all layers |
| `utils-update.awk` | Gradient descent step — SGD, momentum, Adam (with bias correction) |
| `nnet-train.awk` | Training loop orchestration |
| `nnet-predict.awk` | Inference orchestration + metrics output |
| `nnet-init.awk` | Network creation and weight file generation |

## Data Format

**Dataset** (space/tab-separated, `#` comments ignored):
```
# inputs... outputs...
0 0 0
0 1 1
1 0 1
1 1 0
```

**Model layer file** (`models/<name>/layer<N>.txt`):
```
ACTIVATION=sigmoid
w1 w2 ... bias
w1 w2 ... bias
```

## Dependencies

- `gawk` (GNU AWK recommended for full compatibility)
- `bash`
- Standard Unix tools (`mkdir`, `ls`)
