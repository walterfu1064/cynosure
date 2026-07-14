## Index of examples

1. [`zstack_fitting/fitting.ipynb`](zstack_fitting/fitting.ipynb) - Classical phase reconstruction.
   Iteratively fits phase/amplitude Zernike aberrations to a z-stack by differentiable optimization
   of the Debye-Wolf forward model (`ForwardZstack`).
2. [`zstack_decoding/decoding.ipynb`](zstack_decoding/decoding.ipynb) - Learned single-pass estimation.
   Trains a CNN to infer aberration coefficients from a z-stack, using a forwards model bundled into
   the same `ZstackSolver` object to generate synthetic training data on the fly.

## Data

Both examples use the small, real bead patches in
[`bead_patches/`](bead_patches/), cropped from a third-party, open-access
dataset (CC BY 4.0). See [`bead_patches/README.md`](bead_patches/README.md)
for full provenance, attribution, licensing, and crop parameters, and
`data/README.md` at the repo root for obtaining the full (~7 GB) dataset.
