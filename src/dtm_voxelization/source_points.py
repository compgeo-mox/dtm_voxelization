"""Surveyed source points into the DTM's frame, as an .xyz:
`python -m dtm_voxelization.source_points`, or the file itself from an editor.

The survey comes as a CSV of name, Est, Nord, Quota, code in UTM32N. The grid,
the SPEED mesh and the fracture surfaces all live in the DTM's frame, which is
UTM minus the point cloud's mean M (see the README's Data section), so that is
what is taken off here -- z included, as for the cloud itself.

Written next to the survey, in data/source_pts.
"""

import sys
from pathlib import Path

import numpy as np

if __package__:
    from .dtm_io import save_xyz
    from .shift_fracture import M  # the same Rialba mean
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.dtm_io import save_xyz
    from dtm_voxelization.shift_fracture import M

SURVEY = "rialba_source_pts.txt"


def main():
    folder = Path(__file__).resolve().parents[2] / "data" / "source_pts"
    survey = folder / SURVEY
    points = np.loadtxt(survey, delimiter=",", skiprows=1, dtype=float, usecols=(1, 2, 3))
    print(f"read {survey}: {len(points)} points in UTM32N\n{points}")

    points -= M
    print(f"minus the DTM's mean {M}:\n{points}")

    save_xyz(survey.with_suffix(".xyz"), points, f"{SURVEY}, UTM32N minus the DTM's mean")


if __name__ == "__main__":
    main()
