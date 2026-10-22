"""Surveyed source points into the DTM's frame, as an .xyz and a .vtu:
`python -m dtm_voxelization.source_points [CASE.toml]`, or the file itself from
an editor, which takes CASE below.

The survey comes as a CSV of name, Est, Nord, Quota, code in UTM32N. The grid,
the SPEED mesh and the fracture surfaces all live in the DTM's frame, which is
UTM minus the point cloud's mean M (see the README's Data section), so that is
what is taken off here.

The surveyed Quota is then dropped and the points are put on the GRID, not on
the terrain: the grid is what SPEED runs on, and its carved top stands a whole
cell below the terrain in places. Each point lands on the top of the column of
hexes above its Est and Nord, raised by LIFT so that it is outside the rock
rather than exactly on a face, which SPEED then projects back onto the grid --
a source that is unambiguously at the surface.

The surveyed elevations come in about 40 m above the terrain, consistently
(see elevation_offset.md), which means the survey and the DTM do not share a
vertical or a horizontal reference; until that is settled with whoever
measured them, their horizontal position is the part worth trusting. The log
prints both elevations and the drop.

The grid is read from the case's output, so run the `grid` step first; this
does not rebuild it.

Written next to the survey, in data/source_pts: the .xyz for the pipeline, the
.vtu to drop on the grid in ParaView.
"""

import sys
from pathlib import Path

import meshio
import numpy as np

if __package__:
    from .dtm_io import save_xyz
    from .shift_fracture import M  # the same Rialba mean
    from .case import load_case
    from .mesh import read_hex_mesh, require
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.dtm_io import save_xyz
    from dtm_voxelization.shift_fracture import M
    from dtm_voxelization.case import load_case
    from dtm_voxelization.mesh import read_hex_mesh, require

SURVEY = "rialba_source_pts.txt"
CASE = "rialba.toml"  # whose grid the points are dropped onto
LIFT = 1e-3  # m above the grid's top face, so the point is outside the rock
# wanted besides the survey, and already in the DTM's frame: picked off the mesh
# in ParaView rather than measured, so they stay here and out of the survey file
EXTRA = [(-162.0, 248.0)]  # over the first tower


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    survey = root / "data" / "source_pts" / SURVEY
    points = np.loadtxt(survey, delimiter=",", skiprows=1, dtype=float, usecols=(1, 2, 3))
    print(f"read {survey}: {len(points)} points in UTM32N")

    points -= M
    if EXTRA:
        # their z is the grid's, like everyone else's, so anything does here
        points = np.vstack([points, [(x, y, np.nan) for x, y in EXTRA]])
        print(f"  plus {len(EXTRA)} picked off the mesh: {EXTRA}")
    case = load_case(argv[0] if argv else root / "cases" / CASE)
    require(case.grid_path, "grid")
    grid, hexes = read_hex_mesh(case.grid_path)
    corners = grid[hexes]
    lo, hi = corners.min(axis=1), corners.max(axis=1)
    print(f"  {case.grid_path}: {len(hexes):,} hexes")

    top = np.empty(len(points))
    for n, (x, y, _) in enumerate(points):
        over = (lo[:, 0] <= x) & (x <= hi[:, 0]) & (lo[:, 1] <= y) & (y <= hi[:, 1])
        if not over.any():
            raise ValueError(
                f"source point {n + 1} at ({x:.3f}, {y:.3f}) stands over no hex of the grid"
            )
        top[n] = hi[over, 2].max()

    print(f"  {'x':>10} {'y':>10} {'surveyed z':>12} {'grid top':>10} {'dropped by':>12}")
    for (x, y, z), t in zip(points, top):
        print(f"  {x:10.3f} {y:10.3f} {z:12.3f} {t:10.3f} {z - t:12.3f}")
    points[:, 2] = top + LIFT

    where = case.grid_path
    save_xyz(
        survey.with_suffix(".xyz"),
        points,
        f"{SURVEY}, UTM32N minus the DTM's mean, z on {where} + {LIFT} m",
    )
    meshio.write_points_cells(
        survey.with_suffix(".vtu"),
        points,
        [("vertex", np.arange(len(points)).reshape(-1, 1))],
    )
    print(f"wrote {survey.with_suffix('.vtu')}: {len(points)} points to look at")


if __name__ == "__main__":
    main()
