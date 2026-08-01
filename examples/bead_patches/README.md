# Example data

This folder contains small, cropped patches from a third-party, open-access dataset, which are
included so this repo has convenient access to small amounts of real, experimental data.
The full dataset is git-ignored under `data/`, and is **not** redistributed here. See
`data/README.md` for notes on obtaining the original, full dataset (~7 GB).

## Source and attribution

> R. Holinirina Dina Miora, M. Senftleben, S. Abrahamsson, E. Rohwer,
> R. Heintzmann, and G. Bosman, "Experimental validation of numerical point
> spread function calculation including aberration estimation," Opt. Express
> 32(12), 21887–21908 (2024).
> https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-12-21887

Supporting dataset (the actual source of these patches):

> R. H. D. Miora *et al.*, dataset for the above paper, Zenodo (2024).
> DOI: [10.5281/zenodo.10522322](https://doi.org/10.5281/zenodo.10522322)

**License:** the dataset is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. The patches here are redistributed under the same license, with
attribution as above. They are modified from the originals only by cropping
(see below); pixel values are otherwise unchanged. Their inclusion does not imply
endorsement of this project by the original authors.

Everything in this repository *outside* the `examples/` and `data/` folders is
original work under the repository's own license. The CC BY 4.0 terms apply only
to the data patches in this folder.

## What was cropped, and from where

All patches have been hand-selected from the `NA095/` subset of the Zenodo record.
Each source file  is a bead z-stack at `NA095/<acquisition>/stack/stack.tif` within the
record, with axis order `[z, row, col]`. Each patch is a 32x32 pixel spatial crop,
keeping the full z range of the source stack.

| Patch file           | Source z-stack (`NA095/…/stack/stack.tif`) | Approx. bead center (row, col) | Crop (rows, cols)    | Axial step |
|----------------------|--------------------------------------------|--------------------------------|----------------------|------------|
| `bead_patch_01.tiff` | `Z-Stack 13-08-2021 01.15.19`            | (1400, 262)                    | 1384:1416, 246:278   | 0.12 um    |
| `bead_patch_02.tiff` | `Z-Stack 13-08-2021 01.15.19`            | (1407, 596)                    | 1391:1423, 580:612   | 0.12 um    |
| `bead_patch_03.tiff` | `Z-Stack 13-08-2021 01.15.19`            | (780, 1210)                    | 764:796,  1194:1226  | 0.12 um    |
| `bead_patch_04.tiff` | `Z-Stack 13-08-2021 01.15.19`            | (276, 1285)                    | 260:292,  1269:1301  | 0.12 um    |
| `bead_patch_05.tiff` | `Z-Stack 13-08-2021 01.21.20`            | (670, 560)                     | 654:686,  544:576    | 0.13 um    |
| `bead_patch_06.tiff` | `Z-Stack 13-08-2021 01.27.01`            | (1217, 1592)                   | 1201:1233, 1576:1608 | 0.13 um    |


Crop convention (half-open intervals): `stack[:, center_row-16 : center_row+16, center_col-16 : center_col+16]`.

## Imaging parameters (NA095 subset)

Transcribed from `NA095_imaging_parameters.txt` in the Zenodo record:

| Quantity                 | Value                                                      |
|--------------------------|------------------------------------------------------------|
| Microscope               | Thermo Fisher EVOS M5000                                   |
| Objective                | Olympus air PLAN S-APO 40X (AMEP4754), NA 0.95, WD 0.18 mm |
| Camera pixel pitch       | 3.45 um                                                    |
| Object-space pixel pitch | 0.08625 um                                                 |
| Objective focal length   | 4.50 mm (inferred from 180 mm tube lens)                   |
| Emission wavelength      | 0.510 um                                                   |
| Beads                    | TetraSpeck (TS), 0.1 um diameter                           |
| Embedding medium         | Dako, n = 1.3744                                           |

## Reproducing the patches

The patches are fully reproducible from the Zenodo record. Download record
[10522322](https://doi.org/10.5281/zenodo.10522322), place its contents at
`data/10522322/` (git-ignored), and run `extract_patches.py` in this folder,
which reads the crop table above and writes the `bead_patch_*.tiff` files.
