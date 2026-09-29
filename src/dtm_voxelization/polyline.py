"""Every surveyed polyline in data/surfaces/polylines, into the DTM's frame and
out as a .vtu for ParaView: `python -m dtm_voxelization.polyline`, or the file
itself from an editor. Drop a new survey in that folder and it is picked up.

Each file is X Y Z in UTM32N, one point per line under a `//X Y Z` header,
already in order along the line. The grid, the SPEED mesh and the fracture
surfaces live in the DTM's frame, UTM minus the point cloud's mean M (see the
README's Data section), so that is what is taken off -- z included.

A line whose two ends are close compared with its own length is taken as
CLOSED and gets a segment joining the last point back to the first; one whose
ends are far apart is left open. Among the Rialba surveys the outlines close
within 3% of their perimeter while the cavity trace's ends stand a full
length apart, so the two kinds separate cleanly rather than by a guess.

Written beside each survey, as line cells: ParaView draws them as they are.
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

POLYLINES = Path(__file__).resolve().parents[2] / "data" / "surfaces" / "polylines"
CLOSED = 0.1  # ends closer than this much of the line's length: a loop


def paths():
    """Every survey in the polylines folder."""
    found = sorted(POLYLINES.glob("*.txt"))
    if not found:
        raise SystemExit(f"no polyline .txt in {POLYLINES}")
    return found


def is_closed(points):
    """Whether the line comes back to where it started."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    gap = float(np.linalg.norm(points[0] - points[-1]))
    closed = gap < CLOSED * steps.sum()
    print(
        f"  steps along the line: {steps.min():.3f} to {steps.max():.3f} m over "
        f"{steps.sum():.1f} m; the ends stand {gap:.2f} m apart, "
        f"{100 * gap / steps.sum():.0f}% of that -- {'closed' if closed else 'OPEN'}"
    )
    return closed


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

    return points


def main():
    for path in paths():
        points = load(path)
        ends = len(points) if is_closed(points) else len(points) - 1
        segments = np.column_stack([np.arange(ends), np.roll(np.arange(len(points)), -1)[:ends]])
        out = path.with_suffix(".vtu")
        meshio.write_points_cells(out, points, [("line", segments)], binary=True)
        print(f"wrote {out}: {len(segments)} segments")


if __name__ == "__main__":
    main()
