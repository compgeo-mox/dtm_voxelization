"""A surveyed polyline into the DTM's frame, as a .vtu for ParaView:
`python -m dtm_voxelization.polyline [POLYLINE.txt]`, or the file itself from
an editor, which takes POLYLINE below.

The file is the survey's X Y Z in UTM32N, one point per line under a `//X Y Z`
header, already in order along the line. The grid, the SPEED mesh and the
fracture surfaces live in the DTM's frame, UTM minus the point cloud's mean M
(see the README's Data section), so that is what is taken off -- z included.

The line is CLOSED: a segment joins the last point back to the first. The log
prints the step between consecutive points and that closing step beside it, so
a file that is not in order, or not a loop, shows up as one step far longer
than the rest rather than as a quietly wrong shape.

Written next to the survey, as line cells: ParaView draws it as it is.
"""

import sys
from pathlib import Path

import meshio
import numpy as np

if __package__:
    from .shift_fracture import M  # the same Rialba mean
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.shift_fracture import M

POLYLINE = "Polyline_torrione1-2.txt"  # in data/surfaces, unless one is given


def path_of(argv):
    """The polyline named on the command line, or POLYLINE in data/surfaces."""
    if len(argv) > 1:
        raise SystemExit(__doc__)
    if argv:
        return Path(argv[0]).resolve()
    return Path(__file__).resolve().parents[2] / "data" / "surfaces" / POLYLINE


def load(path):
    """The polyline's points in the DTM's frame, in the order it gives them."""
    points = np.loadtxt(path, comments="//", dtype=float)
    print(f"read {path}: {len(points)} points in UTM32N")
    points -= M
    print(
        f"  minus the DTM's mean: x [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] "
        f"y [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] "
        f"z [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]"
    )

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    closing = float(np.linalg.norm(points[0] - points[-1]))
    print(
        f"  steps along the line: {steps.min():.3f} to {steps.max():.3f} m, "
        f"{steps.sum() + closing:.1f} m around; the closing step is {closing:.3f} m"
    )
    if closing > 3 * np.median(steps):
        print(
            f"  WARNING: that closing step is {closing / np.median(steps):.1f} times the "
            f"median one -- the points may not be in order, or the line may not be a loop",
            flush=True,
        )

    return points


def main(argv=None):
    path = path_of(sys.argv[1:] if argv is None else argv)
    points = load(path)
    segments = np.column_stack([np.arange(len(points)), np.roll(np.arange(len(points)), -1)])
    out = path.with_suffix(".vtu")
    meshio.write_points_cells(out, points, [("line", segments)], binary=True)
    print(f"wrote {out}: {len(segments)} segments, the last one closing the loop")


if __name__ == "__main__":
    main()
