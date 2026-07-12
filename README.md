# wave-physics

Tools for modeling beam propagation using Debye-Wolf theory.

Currently built around the use case of taking an unpolarized wavefront at
a microscope objective, aberrating it using a sum of Zernike polynomials,
and propagating it by some distance.

No assumptions about the objective's NA are involved. The major approximations are:
1. that the light is monochromatic (although the results could be summed over reference wavelengths if desired);
2. that evanescent waves are negligible; and
3. that the objective and the propagation distance are both larger than the wavelength.

The framework, however, is more general than that at its core. The machinery can be readily
adapted to work with arbitrarily-polarized beams (which I might expose in the future).

Features I might add in the future:
* exposing arbitrary polarizations
* more complex imaging systems


## Example data credits:
The bead patches in the `examples` folder are cropped from the open-access dataset of
R. H. D. Miora et al., "Experimental validation of numerical point spread function
calculation including aberration estimation," Opt. Express 32(12), pp. 21887-21908 (2024),
https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-12-21887 (last accessed 2026/07/10).
The dataset (Zenodo DOI [10.5281/zenodo.10522322] (https://zenodo.org/records/10522322))
is licensed under CC BY 4.0, and the patches are redistributed here under the same license.
See [`examples/README.md`](examples/README.md) for the exact source files, crop parameters,
and imaging parameters.


## License

The source code and documentation in this repository are licensed under the
[MIT License](LICENSE). The example data patches in [`examples/`](examples/) are
derived from the third-party dataset above and are redistributed under CC BY 4.0;
see [`examples/README.md`](examples/README.md) for full attribution.