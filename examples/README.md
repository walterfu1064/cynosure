## Index of examples

1. [`zstack_fitting/example_1.ipynb`](zstack_fitting/example_1.ipynb) - **Classical phase reconstruction.**
   We start with the Debye-Wolf forwards simulation model, showing how it can be used to iteratively fit
   aberrations in experimentally-obtained images of small beads. Using differentiable optimization, the
   model iteratively fits phase/amplitude Zernike aberrations to example z-stack images.
2. [`zstack_decoding`](zstack_decoding/) - **Learned, amortized estimation.**
   We use the forwards simulation model to generate synthetic images labeled with their aberrations,
   and use it to train various decoders to predict those aberrations in a single pass. Each example
   models the posterior more thoroughly than the last.
   1. [`zstack_decoding/example_2a.ipynb`](zstack_decoding/example_2a.ipynb) - Point estimation of the most probable aberration coefficients.
   2. [`zstack_decoding/example_2b.ipynb`](zstack_decoding/example_2b.ipynb) - Estimation of the most probable aberration coefficients
      and the univariate uncertainty of each.
   3. [`zstack_decoding/example_2c.ipynb`](zstack_decoding/example_2c.ipynb) - Estimation of the most probable aberration coefficients
      and their pairwise covariance matrix.
   4. [`zstack_decoding/example_2d.ipynb`](zstack_decoding/example_2d.ipynb) - Estimation of a full mixture density over the aberration
      coefficients. This is the first model that can represent a multimodal posterior, and we take advantage
      of that by moving away from z-stacks, and towards single-image inference, where phase conjugation
      produces a known degeneracy across the focal plane.
   5. [`zstack_decoding/example_2e.ipynb`](zstack_decoding/example_2e.ipynb) - Estimation of the same single-image posterior by flow
      matching, allowing us to represent arbitrary posterior distributions, albeit not in closed form.
   6. [`zstack_decoding/example_2f.ipynb`](zstack_decoding/example_2f.ipynb) - Set transformer encoder with cross-z
      attention pooling to allow z-stack geometry (number, placement, and ordering of planes) to vary dynamically.
      Comparison of the fitted attention weights to the mutual information vs. z.
3. [`information_theory`](information_theory/) - **Sandbox for exploring information content in aberrated
   images**. Makes use of the aforementioned models. Will probably continue to evolve as ideas strike my fancy.
   1. [`information_theory/example_3a.ipynb`](information_theory/example_3a.ipynb) - Analysis of how the
      predicted uncertainty in each coefficient varies as a function of defocus, and comparing the
      model's uncertainty against the fundamental van Trees limit. Also, calibration of the trained
      posterior using SBC.
   2. [`information_theory/example_3b.ipynb`](information_theory/example_3b.ipynb) - Analysis of information in
      the forwards model by itself, with no trained network involved. Playing with the results leads to the neat
      observation that the total available information peaks roughly one depth-of-field from the focal plane.
   3. [`information_theory/example_3c.ipynb`](information_theory/example_3c.ipynb) - SBC calibration of the
      flow-matching model, analogous to `example_3a.ipynb`. The defocus term appears stubbornly underconfident,
      and remains an open question.
4. [`object_decoding`](object_decoding) - **Simultaneously learning aberrations and an unknown object.**
   1. [`object_decoding/example_4a.ipynb`](object_decoding/example_4a.ipynb) - Extension of `example_3a.ipynb`
          to vary the bead diameter too, using a model trained with randomly sampled beads.
   2. [`object_decoding/example_4b.ipynb`](object_decoding/example_4b.ipynb) - Extension of `example_2e.ipynb`
      to jointly predict aberrations and a more complicated, extended object (two blobs of differing diameters,
      intensities, and positions).

## Data

Example 1 fits the small, real bead patches in
[`bead_patches/`](bead_patches/), cropped from a third-party, open-access
dataset (CC BY 4.0). See [`bead_patches/README.md`](bead_patches/README.md)
for full provenance, attribution, licensing, and crop parameters, and
`data/README.md` at the repo root for obtaining the full (~7 GB) dataset.

The decoding examples need no stored data at all, since each trains against synthetic images that its own
forwards model generates on the fly.
