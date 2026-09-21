"""Surveyed source points into the DTM's frame, as an .xyz:
`python -m dtm_voxelization.source_points`, or the file itself from an editor.

The survey comes as a CSV of name, Est, Nord, Quota, code in UTM32N. The grid,
the SPEED mesh and the fracture surfaces all live in the DTM's frame, which is
UTM minus the point cloud's mean M (see the README's Data section), so that is
what is taken off here.

The surveyed Quota is then dropped and the points are put ON the terrain, at
the case's DTM interpolated at their Est and Nord. They come in about 40 m
above it, consistently (see elevation_offset.md), which means the survey and
the DTM do not share a vertical or a horizontal reference; until that is
settled with whoever measured them, their horizontal position is the part
worth trusting. The log prints how far each point fell.

Written next to the survey, in data/source_pts.
"""

import sys
from pathlib import Path

import numpy as np

if __package__:
    from .dtm_io import save_xyz
    from .shift_fracture import M  # the same Rialba mean
    from .case import load_case
    from . import terrain as terrain_surfaces
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.dtm_io import save_xyz
    from dtm_voxelization.shift_fracture import M
    from dtm_voxelization.case import load_case
    from dtm_voxelization import terrain as terrain_surfaces

SURVEY = "rialba_source_pts.txt"
CASE = "rialba.toml"  # the DTM the points are dropped onto


def main():
    root = Path(__file__).resolve().parents[2]
    survey = root / "data" / "source_pts" / SURVEY
    points = np.loadtxt(survey, delimiter=",", skiprows=1, dtype=float, usecols=(1, 2, 3))
    print(f"read {survey}: {len(points)} points in UTM32N")

    points -= M
    terrain, _, _ = terrain_surfaces.build(load_case(root / "cases" / CASE))
    ground = terrain.interp(points[:, :2])
    print(f"  {'x':>10} {'y':>10} {'surveyed z':>12} {'terrain':>10} {'dropped by':>12}")
    for (x, y, z), g in zip(points, ground):
        print(f"  {x:10.3f} {y:10.3f} {z:12.3f} {g:10.3f} {z - g:12.3f}")
    points[:, 2] = ground

    save_xyz(
        survey.with_suffix(".xyz"),
        points,
        f"{SURVEY}, UTM32N minus the DTM's mean, z from the {CASE} terrain",
    )


if __name__ == "__main__":
    main()
