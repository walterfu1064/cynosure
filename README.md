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
* phase retrieval and aberration fitting