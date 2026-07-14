# Local data directory (not redistributed)

This directory holds the full third-party dataset used for development and for
generating the example patches. Its contents are git-ignored and are **not**
redistributed in this repository -- only this README is tracked. The small cropped
bead patches that ship with the repo live in
[`../examples/bead_patches/`](../examples/bead_patches/). See
[`../examples/bead_patches/README.md`](../examples/bead_patches/README.md) for their
full provenance, attribution, and crop parameters.

## Getting the dataset

Download Zenodo record
[10.5281/zenodo.10522322](https://doi.org/10.5281/zenodo.10522322) (Miora *et al.*,
"Experimental validation of numerical point spread function calculation including
aberration estimation," Opt. Express 32(12), 21887–21908, 2024; licensed
CC BY 4.0) and unpack it here so the layout is:

```
data/10522322/NA095/<acquisition>/stack/stack.tif
```

Then, from the repo root, regenerate the committed example patches with:

```
python examples/bead_patches/extract_patches.py
```
