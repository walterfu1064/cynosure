# cynosure

> **cynosure** (*n.*) - a focal point; something that guides. From Greek κυνόσουρα,
> "dog's tail," the old name for Ursa Minor, whose tail-tip is the guide star Polaris. Dogs are good.

Microscopy beam propagation and optical aberration modeling using Debye-Wolf theory and machine learning.

### Forward model

The core is a fully differentiable beam propagation module that takes a wavefront at a microscope
objective, aberrates it using a sum of Zernike polynomials, and propagates it some distance.

The propagation code makes few assumptions about the imaging system. The major approximations are:
- No restrictions are placed on the objective's NA (apart from it being no greater than the immersion medium's index)
- The light is assumed to be monochromatic (although the results could be summed over a spectrum if desired)
- Evanescent waves are assumed to be negligible
- The focal length and the propagation distance are both assumed to be much larger than the wavelength


### Inverting the forward model

A family of models wraps the propagator to run it backwards, using measured (or simulated) images of a
known emitter to recover the pupil aberrations that produced them. Two major approaches are currently implemented.

**Per-stack fitting (classical phase retrieval).** Since the forward model is end-to-end differentiable, the
aberrations can be fitted using gradient descent against an observed z-stack. The optimization must be done
anew for each stack, but requires no training and no prior. Aberrations are parameterized in the Zernike basis,
so the fit is directly interpretable as the Seidel aberrations (defocus, astigmatism, coma, etc.).

**Amortized inference (learned decoder).** For a given optical system, the forward model doubles as a simulator,
allowing a decoder (a CNN, e.g.) to be trained on synthetic (image, coefficient) pairs generated on the fly.
As an amortized simulation-based inference approach, once the expensive training is done, the decoder can subsequently
predict aberration coefficients in a single pass on image data from the asme optical system.

Several decoders are implemented, which model the posterior in increasingly thorough ways:
- **Point estimate.** Just the conditional means of the aberration coefficients.
- **Heteroscedastic.** A separate mean and variance for each coefficient (i.e., a diagonal Gaussian posterior).
- **Full covariance.** Cholesky factor of a joint Gaussian, capturing the pairwise covariances among the coefficients.
- **Mixture density.** A Gaussian mixture over the coefficients, which can also represent multimodality.

The same machinery can be used to train on either full z-stacks or on single-z images. For the former, the
relative z-positions of the z-stack should be given. For the latter, a z-jitter parameter defines the span of
z-positions generatively modeled, which should match the z-range one hopes to decode.

Worked examples of all of the above are in [`examples/`](examples/README.md).


### Implementation notes

- The pupil-to-object-plane transform is a chirp-Z transform rather than an FFT so the object- and image-plane
  grids can be decoupled for efficient computation.
- The Zernike basis is truncated out to a given maximum radial order. If the system's symmetries are known in
  advance, they can be further restricted to some subset of `(n, m)` indices.
- Where applicable, Zernike coefficients are drawn from a prior that decays as a power of the radial order.
  This is a convenience, not something based in any underlying physics. During training, regression targets
  are whitened against this decay to keep them of magnitude order 1.
- Simulation and physical parameters are defined using a set of config dataclasses. Dimensional parameters
  such as the wavelength or focal length may be passed in any units as long as they are mutually consistent.
  It is assumed that the user won't make a silly choice that tanks numerical stability.

These wrapper models are implemented for unpolarized light. However, the core beam propagator is fully
vectorial and supports arbitrary polarizations. Future work may include wiring that through to the models.


### Example data credits
The bead patches in the `examples/bead_patches` folder are cropped from the open-access dataset of
R. H. D. Miora et al., "Experimental validation of numerical point spread function
calculation including aberration estimation," Opt. Express 32(12), pp. 21887-21908 (2024),
https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-12-21887 (last accessed 2026/07/10).
The dataset (Zenodo DOI [10.5281/zenodo.10522322] (https://zenodo.org/records/10522322))
is licensed under CC BY 4.0, and the patches are redistributed here under the same license.
See [`examples/bead_patches/README.md`](examples/bead_patches/README.md) for the exact
source files, crop parameters, and imaging parameters.


### License

The source code and documentation in this repository are licensed under the
[MIT License](LICENSE). The example data patches in
[`examples/bead_patches/`](examples/bead_patches/) are derived from the third-party
dataset above and are redistributed under CC BY 4.0; see
[`examples/bead_patches/README.md`](examples/bead_patches/README.md) for full attribution.