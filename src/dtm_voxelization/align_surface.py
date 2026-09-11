"""Bring an externally built surface into this project's coordinate frame.

`data/xyz/merged.xyz` is not in map coordinates. `to_xyz.py` (in the
CubitPython4SPEED Rialba folder) built it from the UTM32N DTM by
subtracting the mean of ALL THREE coordinate columns, so the frame used
here is

    (x, y, z)_here = (x, y, z)_UTM32N - MEAN_COORD

with MEAN_COORD below, recomputed from `DTMRialba5m_NEWUTM32N.csv`. The
x/y half of that shift is the one everybody remembers; the z half is easy
to miss, and a surface that kept its absolute elevation ends up ~475 m in
the air, entirely inside the space the terrain carving removes -- which is
exactly what happened to `frattura_verticale_shifted.stl`: its x/y were
shifted to (within 2.6 m of) this origin, its z was not.

Each entry of `SOURCES` names an STL and the translation that lands it in
this frame; the aligned copy is written to `output/planes/`, where
`cut_surface.py` picks it up.
"""

from pathlib import Path

import numpy as np
import meshio

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output" / "planes"
GRID_PATH = REPO_ROOT / "output" / "cartgrid_carved_refined.vtu"

# mean of DTMRialba5m_NEWUTM32N.csv (x_newUTM32N, y_newUTM32N, Z), the
# origin to_xyz.py centred merged.xyz on
MEAN_COORD = np.array([527837.392605, 5082191.470488, 474.976610])

SPEED_SCRIPTS = Path(
    "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo"
    "/codes/speed-repo/scripts"
)

SOURCES = [
    {
        "name": "frattura_verticale",
        "path": SPEED_SCRIPTS / "frattura_verticale_shifted.stl",
        # x/y already carry the shift (to within 2.6 m of MEAN_COORD's own
        # origin, see module docstring); only the elevation is still absolute
        "translation": (0.0, 0.0, -MEAN_COORD[2]),
    },
]


def align(source):
    """Read one source STL, translate it, return the meshio mesh."""
    mesh = meshio.read(str(source["path"]))
    mesh.points = mesh.points.astype(float) + np.asarray(
        source["translation"], dtype=float
    )
    return mesh


def main():
    grid_bbox = None
    if GRID_PATH.exists():
        grid = meshio.read(str(GRID_PATH))
        grid_bbox = (grid.points.min(axis=0), grid.points.max(axis=0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        mesh = align(source)
        out_path = OUTPUT_DIR / f"{source['name']}.stl"
        meshio.write(str(out_path), mesh)

        lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
        print(f"wrote {out_path}")
        print(f"  from {source['path'].name}, translated by {source['translation']}")
        print(f"  bbox {lo.round(2)} .. {hi.round(2)}")
        if grid_bbox is not None:
            inside = np.all((hi >= grid_bbox[0]) & (lo <= grid_bbox[1]))
            print(f"  overlaps the grid's bounding box: {bool(inside)}", flush=True)


if __name__ == "__main__":
    main()
