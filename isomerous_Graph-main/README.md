# isomerous Graph

This repository implements a modular pipeline for building multi-view graphs
from resting-state fMRI data and training a Graph Transformer with
mixture-of-experts edge gating.  The code is organised into packages that mirror
the end-to-end workflow – from preprocessing to interpretability.

## Directory layout

```
project/
  configs/
  src/
  scripts/
```

The code inside `project/src` implements a minimal but functional pipeline for
constructing edge features with a mixture-of-experts routing mechanism, running
a multi-relation graph transformer, estimating communities and producing a
classifier head.

## Smoke test

To verify that the model can execute end-to-end using random data, run:

```bash
python project/scripts/train.py
```

This script instantiates `BrainGraphModel`, performs a single forward pass on a
randomly generated batch and prints the aggregated losses to confirm that
`BrainGraphModel.forward(batch)` produces both logits and auxiliary outputs.

The remainder of the repository is organised to make it easy to replace the
simplified components with production-grade implementations.
