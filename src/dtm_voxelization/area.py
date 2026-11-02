"""A wall's reconstructed surface, whole and cut to the area of interest:
`python -m dtm_voxelization.area [CASE.toml [CLOUD.las]]`, or the file itself
from an editor, which takes CASE below.

With a cloud, the surface is built from all of it -- the whole wall, which is
the point of passing one, the case's own points being already cut. Without,
from the case's points.

The area is drawn on the wall seen FROM THE FRONT, which is how one picks a
piece of something sub-vertical, so its file is a closed polygon of X and Z in
UTM32N -- two columns, whatever its header says. The northing would tell
nothing: what the polygon describes is a prism running straight through the
rock along y. The case's `origin` says where the cloud's own zero sits in
UTM32N, and only the easting comes off; the cloud's z is already an elevation.

A point is inside when its x and z fall in the polygon, tested in the cloud's
own frame. The grid is built in another one -- `frame.py` turns the wall's
least-squares plane horizontal, which at San Martino sends the elevation onto
the grid's x and the wall's normal onto its z -- and in that frame the prism
comes out leaning by some ten degrees, so it is no box to clip against. Going
back to the cloud's frame costs one matrix and is exact.

Writes, in the case's output and all three in the CLOUD's frame -- x and y
centred on the case's origin, z the elevation -- not in the grid's. The window
is drawn in that frame, and these three are for looking at it:

- wall.stl: the whole reconstruction.
- wall_in_area.stl: the same with only the triangles whose centre falls in the
  area, which is the piece the grid should be built on.
- area_of_interest.stl: the prism itself, as the band its walls make across
  the surface's own span of y, closed at both ends.
"""

import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

if __package__:
    import laspy

    from . import frame
    from .case import load_case
    from .dtm_io import _load_xyz
    from .terrain import export_triangles, poisson_surface
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import laspy

    from dtm_voxelization import frame
    from dtm_voxelization.case import load_case
    from dtm_voxelization.dtm_io import _load_xyz
    from dtm_voxelization.terrain import export_triangles, poisson_surface

CASE = "cases/san_martino.toml"  # when none is given, as from an editor's Run button


def window(case):
    """The area's polygon, (x, z) in the cloud's own frame."""
    if case.area is None:
        raise SystemExit(f"{case.name}: no [dtm] area, so no window to cut")
    if case.origin is None:
        raise SystemExit(f"{case.name}: [dtm] area needs [dtm] origin to place it")
    corners = np.loadtxt(case.area, skiprows=1)
    if corners.shape[1] != 2:
        raise SystemExit(
            f"{case.area}: {corners.shape[1]} columns, expected 2 -- x and z, "
            "the area being drawn on the wall seen from the front"
        )
    corners[:, 0] -= case.origin[0]
    if np.allclose(corners[0], corners[-1]):
        corners = corners[:-1]
    print(
        f"  {case.area.name}: {len(corners)} corners, x "
        f"[{corners[:, 0].min():.1f},{corners[:, 0].max():.1f}] z "
        f"[{corners[:, 1].min():.1f},{corners[:, 1].max():.1f}]",
        flush=True,
    )
    return corners


def inside(corners, points):
    """Which of the points, in the cloud's own frame, fall in the window."""
    shape = Polygon(corners).buffer(0)
    shapely.prepare(shape)
    return shapely.contains_xy(shape, points[:, 0], points[:, 2])


def prism(corners, reach):
    """(points, triangles) of the window swept along y over `reach`.

    The walls are a band of two triangles per corner, and each end is fanned
    from the first corner -- exact for the quadrilateral an area of interest
    usually is, and the ends are there only to show which way the prism runs."""
    near = np.column_stack([corners[:, 0], np.full(len(corners), reach[0]), corners[:, 1]])
    far = np.column_stack([corners[:, 0], np.full(len(corners), reach[1]), corners[:, 1]])
    points = np.vstack([near, far])
    n = len(corners)
    walls = [
        tri
        for k in range(n)
        for tri in ((k, (k + 1) % n, n + k), ((k + 1) % n, n + (k + 1) % n, n + k))
    ]
    caps = [(0, k, k + 1) for k in range(1, n - 1)]
    caps += [(n, n + k + 1, n + k) for k in range(1, n - 1)]
    return points, np.array(walls + caps)


def cloud_of(case, path):
    """The points to build on, in the cloud's frame: a LAS as it comes, minus
    the case's origin, or else the case's own points, already in that frame."""
    if path is None:
        x, y, z = _load_xyz(case.points)
        points = np.column_stack([x, y, z])
        print(f"  {case.points.name}: {len(points):,} points", flush=True)
        return points
    las = laspy.read(path)
    points = np.column_stack([las.x, las.y, las.z])
    points[:, :2] -= case.origin
    print(
        f"  {Path(path).name}: {len(points):,} points, minus the case's origin: "
        f"x [{points[:, 0].min():.1f},{points[:, 0].max():.1f}] "
        f"y [{points[:, 1].min():.1f},{points[:, 1].max():.1f}] "
        f"z [{points[:, 2].min():.1f},{points[:, 2].max():.1f}]",
        flush=True,
    )
    return points


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 2:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)
    corners = window(case)
    points = cloud_of(case, argv[1] if len(argv) > 1 else None)

    # the reconstruction wants the wall lying down, its normals seeded upwards;
    # what comes out is turned back, so all three files share the cloud's frame
    surface, rotation, center = poisson_surface(points, case)
    vertices = frame.to_dtm(surface.vertices, rotation, center)

    case.output.mkdir(parents=True, exist_ok=True)
    export_triangles(case.output / "wall.stl", vertices, surface.triangles)
    print(f"wrote {case.output / 'wall.stl'}: the whole wall")

    keep = inside(corners, vertices[surface.triangles].mean(axis=1))
    print(
        f"  {int(keep.sum()):,} of {len(surface.triangles):,} triangles have their centre "
        f"in the area ({100 * keep.mean():.1f}%)",
        flush=True,
    )
    export_triangles(case.output / "wall_in_area.stl", vertices, surface.triangles, keep=keep)
    print(f"wrote {case.output / 'wall_in_area.stl'}: the piece inside the window")

    span = (vertices[:, 1].min(), vertices[:, 1].max())
    walls, triangles = prism(corners, span)
    export_triangles(case.output / "area_of_interest.stl", walls, triangles)
    print(
        f"wrote {case.output / 'area_of_interest.stl'}: the prism, drawn across "
        f"y {span[0]:.1f} to {span[1]:.1f}"
    )


if __name__ == "__main__":
    main()
