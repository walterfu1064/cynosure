"""
Extracts the small, example image patches referenced in this repository.

The full dataset (Zenodo record 10522322, CC BY 4.0) is not redistributed in this repository.
This script regenerates the small `bead_patch_*.tiff` files in this folder from that dataset, so
the crops are fully reproducible. See `README.md` in this folder for provenance and attribution.

The resulting patches are already committed in this folder, so this helper script does not need
to be run to access them. Rather, it is included for reproducibility and to clarify data provenance.

Usage:
    Download https://doi.org/10.5281/zenodo.10522322 and unpack it to
    `data/10522322/` at the repo root, then run:

        uv run python examples/bead_patches/extract_patches.py
"""

from dataclasses import dataclass
from pathlib import Path

import tifffile

PATCHES_DIR = Path(__file__).resolve().parent
REPO_ROOT = PATCHES_DIR.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "10522322" / "NA095"
HALF_WIDTH = 16  # half-width of the generated crops

@dataclass(frozen=True)
class Patch:
    output_name: str
    zstack_folder: str
    center_row: int
    center_column: int
    half_width: int


# See README.md for the full provenance table
PATCHES = [
    Patch("bead_patch_01.tiff", "Z-Stack 13-08-2021 01.15.19", 1400, 262, HALF_WIDTH),
    Patch("bead_patch_02.tiff", "Z-Stack 13-08-2021 01.15.19", 1407, 596, HALF_WIDTH),
    Patch("bead_patch_03.tiff", "Z-Stack 13-08-2021 01.15.19", 780, 1210, HALF_WIDTH),
    Patch("bead_patch_04.tiff", "Z-Stack 13-08-2021 01.15.19", 276, 1285, HALF_WIDTH),
    Patch("bead_patch_05.tiff", "Z-Stack 13-08-2021 01.21.20", 670, 560, HALF_WIDTH),
    Patch("bead_patch_06.tiff", "Z-Stack 13-08-2021 01.27.01", 1217, 1592, HALF_WIDTH),
]


def main():
    # for out_name, acquisition, h0, w0 in PATCHES:
    for patch in PATCHES:
        stack_path = DATA_ROOT / patch.zstack_folder / "stack" / "stack.tif"
        if not stack_path.exists():
            raise FileNotFoundError(
                f"Source stack not found: {stack_path}\n"
                "Download Zenodo record 10522322 to data/10522322/ first."
            )
        stack = tifffile.imread(stack_path)  # [z, row, col]

        rows = slice(patch.center_row - patch.half_width, patch.center_row + patch.half_width)
        cols = slice(patch.center_column - patch.half_width, patch.center_column + patch.half_width)
        crop = stack[:, rows, cols]
        output_path = PATCHES_DIR / patch.output_name
        tifffile.imwrite(output_path, crop)
        print(f"wrote {output_path.name} shape={crop.shape} dtype={crop.dtype}")


if __name__ == "__main__":
    main()
