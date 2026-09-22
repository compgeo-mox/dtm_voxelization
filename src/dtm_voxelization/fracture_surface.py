"""A triangulated surface bounded by a surveyed polyline, as an STL a case can
cut with: `python -m dtm_voxelization.fracture_surface [POLYLINE.txt]`, or the
file itself from an editor.

The polyline is not planar, and the surface keeps its shape. Three steps:

- MESH IN THE PLANE. The polyline's least-squares plane gives (u, v) to
  triangulate in and w, the distance off it, to put back afterwards. The
  polygon is filled with a regular lattice at the polyline's own median step,
  Delaunay-triangulated together with the boundary points, and the triangles
  whose centre falls outside the polygon are dropped -- which is what keeps
  the re-entrant parts of this outline empty instead of bridged.

- THE SURFACE ITSELF. w is fitted over (u, v) by a thin-plate spline through
  the polyline's own points: the smoothest surface that passes through every
  one of them exactly. That is the analytic surface the mesh then sits on --
  the interior vertices get their w from it rather than lying flat on the
  plane.

- EXPAND. The outline is grown by EXPAND about its centre in the plane, so the
  surface reaches a little past the survey. Growing it downwards would be
  wrong -- the foot of the polyline is where the fracture was actually seen to
  end -- so whatever the growth pushes below the polyline's own lowest point is
  brought back up to it, flattening the bottom edge at that level and leaving
  the rest alone. Past the original outline the spline extrapolates, which is
  why EXPAND is a few per cent and not a factor. The surface keeps passing
  through the polyline itself, which now lies just inside its boundary.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree, Delaunay

if __package__:
    from . import frame
    from .polyline import load, path_of
    from .terrain import export_triangles
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization import frame
    from dtm_voxelization.polyline import load, path_of
    from dtm_voxelization.terrain import export_triangles

EXPAND = 0.05  # of the outline's size, about its centre in the plane
OUTWARD = (0.0, 1.0, 0.0)  # which way the surface faces, for the plane's normal


def inside(polygon, points):
    """Crossing-number test: which points are inside the closed polygon."""
    x, y = points[:, 0], points[:, 1]
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    straddles = (y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = (x2 - x1)[None, :] * (y[:, None] - y1[None, :]) / (y2 - y1)[None, :] + x1[None, :]
    return (straddles & (x[:, None] < crossing)).sum(axis=1) % 2 == 1


def mesh_polygon(outline, step, keep_as_vertices):
    """(points, triangles) filling the closed polygon `outline` (N, 2), with a
    lattice of interior points at `step` and no triangle outside it.
    `keep_as_vertices` are points that must end up in the mesh -- the survey's
    own, which the expanded outline leaves inside -- and the lattice keeps its
    distance from them so they do not spawn slivers."""
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    grid = np.meshgrid(
        np.arange(lo[0] + step / 2, hi[0], step),
        np.arange(lo[1] + step / 2, hi[1], step),
        indexing="ij",
    )
    lattice = np.column_stack([g.ravel() for g in grid])
    lattice = lattice[inside(outline, lattice)]
    far = cKDTree(keep_as_vertices).query(lattice)[0] > step / 2
    lattice = lattice[far]
    points = np.vstack([outline, keep_as_vertices, lattice])
    triangles = Delaunay(points).simplices
    keep = inside(outline, points[triangles].mean(axis=1))
    print(
        f"  {len(outline)} outline points, {len(keep_as_vertices)} survey points and "
        f"{len(lattice):,} lattice ones at {step:.2f} m: "
        f"{int(keep.sum()):,} triangles kept of {len(triangles):,} (the rest fall outside)",
        flush=True,
    )
    return points, triangles[keep]


def main(argv=None):
    path = path_of(sys.argv[1:] if argv is None else argv)
    points = load(path)

    rotation, center = frame.compute(points, OUTWARD)
    local = frame.to_grid(points, rotation, center)
    print(
        f"  least-squares plane: normal {rotation[2].round(4)}, the polyline stands off it "
        f"by {local[:, 2].std():.3f} m rms, {np.abs(local[:, 2]).max():.3f} m at most"
    )

    outline = local[:, :2] * (1.0 + EXPAND)
    step = float(np.median(np.linalg.norm(np.diff(local[:, :2], axis=0), axis=1)))
    mesh, triangles = mesh_polygon(outline, step, local[:, :2])

    # the analytic surface: the smoothest w(u, v) through the polyline itself
    surface = RBFInterpolator(local[:, :2], local[:, 2], kernel="thin_plate_spline")
    vertices = frame.to_dtm(np.column_stack([mesh, surface(mesh)]), rotation, center)
    at_boundary = np.abs(surface(local[:, :2]) - local[:, 2]).max()
    print(f"  thin-plate spline through the polyline, off it by at most {at_boundary:.2e} m")

    floor = points[:, 2].min()
    below = vertices[:, 2] < floor
    print(
        f"  {int(below.sum())} vertices grew below the polyline's floor of {floor:.2f} m, "
        f"by up to {(floor - vertices[below, 2]).max() if below.any() else 0.0:.2f} m: "
        f"brought back up to it"
    )
    vertices[below, 2] = floor
    print(
        f"  grown by {100 * EXPAND:g}%, floor held: "
        f"x [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}] "
        f"y [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}] "
        f"z [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]"
    )
    export_triangles(path.with_suffix(".stl"), vertices, triangles)


if __name__ == "__main__":
    main()
