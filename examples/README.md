## Index of examples

1. [`zstack_fitting/example_1.ipynb`](zstack_fitting/example_1.ipynb) - Classical phase reconstruction.
   Iteratively fits phase/amplitude Zernike aberrations to a z-stack by differentiable optimization
   of the Debye-Wolf forward model (`ForwardZstack`).
2. [`zstack_decoding`](zstack_decoding/) - Learned single-pass estimation.
   Trains a CNN to infer aberration coefficients from a z-stack, using a forwards model bundled into
   the same `ZstackSolver` object to generate synthetic training data on the fly.
   1. [`zstack_decoding/example_2a.ipynb`](zstack_decoding/example_2a.ipynb) - Point estimation of the
          most probable aberrations coefficients.
   2. [`zstack_decoding/example_2b.ipynb`](zstack_decoding/example_2b.ipynb) - Estimation of the
          most probable aberrations coefficients and the univariate uncertainty of each.
   3. [`zstack_decoding/example_2c.ipynb`](zstack_decoding/example_2c.ipynb) - Estimation of the
          most probable aberrations coefficients and their pairwise covariance matrix.
   4. [`zstack_decoding/example_2d.ipynb`](zstack_decoding/example_2d.ipynb) - Estimation of a full
          mixture density over the aberration coefficients, from a single image rather than a z-stack.
          Trains under a jittered focus position, and resolves the resulting bimodal posterior.

## Data

Example 1 fits the small, real bead patches in
[`bead_patches/`](bead_patches/), cropped from a third-party, open-access
dataset (CC BY 4.0). See [`bead_patches/README.md`](bead_patches/README.md)
for full provenance, attribution, licensing, and crop parameters, and
`data/README.md` at the repo root for obtaining the full (~7 GB) dataset.

The decoding examples need no stored data at all, since each trains against synthetic images that its own
forwards model generates on the fly.
