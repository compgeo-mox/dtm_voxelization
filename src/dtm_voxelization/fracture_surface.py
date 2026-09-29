"""A triangulated surface bounded by each closed polyline of
data/surfaces/polylines, as the STL a case cuts with:
`python -m dtm_voxelization.fracture_surface`, or the file itself from an
editor.

An OPEN line bounds nothing, so it is not a surface of its own. It is a
GUIDE instead: a trace surveyed across the inside of a cavity, saying where
the surface runs between its edges. An open line is taken as a guide to a closed outline when it runs inside it
AND stays within the band that outline itself occupies off its own plane,
doubled. Both tests are needed: the cavity's trace also falls inside the
outline of tower 1-2 seen in that plane, but 21 m behind it, while it sits
0.2 m off the cavity's own band. The surface then passes through the guide as
exactly as through the outline -- without it, the cavity's surface missed its
own internal trace by 1.6 m on average and 2.8 m at worst.

The polyline is not planar, and the surface keeps its shape. Three steps:

- MESH IN THE PLANE. The polyline's least-squares plane gives (u, v) to
  triangulate in and w, the distance off it, to put back afterwards. The
  polygon is filled with a regular lattice at the polyline's own median step,
  Delaunay-triangulated together with the boundary points, and the triangles
  whose centre falls outside the polygon are dropped -- which is what keeps
  the re-entrant parts of this outline empty instead of bridged.

- THE SURFACE ITSELF. w is fitted over (u, v) by a thin-plate spline through
  the polyline's own points and any guide's: the smoothest surface that passes
  through every one of them exactly. That is the analytic surface the mesh then sits on --
  the interior vertices get their w from it rather than lying flat on the
  plane.

- EXPAND. The outline is pushed outwards by EXPAND of its own size, so the
  surface reaches a little past the survey. It is an offset, not a scaling
  about the centre: scaling swings a long closing chord across the corner it
  cuts, which on two of the Rialba outlines left the polyline's first point
  OUTSIDE the surface, by 1.5 and 2.9 m. An outward offset contains the
  original outline by construction. Growing it downwards would be
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
from shapely import Polygon

if __package__:
    from . import frame
    from .polyline import is_closed, load, paths
    from .terrain import export_triangles
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization import frame
    from dtm_voxelization.polyline import is_closed, load, paths
    from dtm_voxelization.terrain import export_triangles

EXPAND = 0.01  # of the outline's size, pushed outwards from it
REFINE = 1  # triangles per step of the survey, along each direction: the mesh
# is built at the polyline's own median step divided by this, so 2 gives about
# four times the triangles and 3 about nine
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


def offset(outline, distance):
    """The closed outline pushed `distance` outwards, all of it, with the
    corners mitred rather than rounded so straight edges stay straight."""
    grown = Polygon(outline).buffer(distance, join_style="mitre", quad_segs=1)
    if grown.geom_type != "Polygon" or grown.interiors:
        raise RuntimeError(f"offsetting the outline gave a {grown.geom_type} with holes")
    return np.asarray(grown.exterior.coords)[:-1]


def resample(outline, step):
    """The closed outline walked round at `step`, so the boundary is as fine as
    the inside. The survey's own points go back in as vertices separately, so
    nothing measured is lost between two of these."""
    closed = np.vstack([outline, outline[:1]])
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    wanted = np.arange(0.0, walked[-1], step)
    return np.column_stack([np.interp(wanted, walked, closed[:, k]) for k in (0, 1)])


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
    kept = triangles[keep]
    lost = ~np.isin(np.arange(len(outline), len(outline) + len(keep_as_vertices)), kept)
    if lost.any():
        print(
            f"  WARNING: {int(lost.sum())} of the {len(keep_as_vertices)} survey points ended "
            f"up in no kept triangle, so the polyline no longer lies on the surface there -- "
            f"they sit nearer the outline than a triangle is wide. Raise EXPAND (a wider "
            f"corona) or lower REFINE (wider triangles).",
            flush=True,
        )
    return points, kept


def surface_of(path, points, traces):
    """Write the STL of the surface bounded by one closed polyline, passing
    through whichever of `traces` runs inside it."""
    rotation, center = frame.compute(points, OUTWARD)
    local = frame.to_grid(points, rotation, center)
    print(
        f"  least-squares plane: normal {rotation[2].round(4)}, the polyline stands off it "
        f"by {local[:, 2].std():.3f} m rms, {np.abs(local[:, 2]).max():.3f} m at most"
    )

    step = float(np.median(np.linalg.norm(np.diff(local[:, :2], axis=0), axis=1))) / REFINE
    size = float(np.ptp(local[:, :2], axis=0).max())
    outline = resample(offset(local[:, :2], EXPAND * size), step)
    print(f"  outline pushed out by {EXPAND * size:.2f} m, {100 * EXPAND:g}% of its {size:.0f} m")

    data = local
    spread = float(np.ptp(local[:, 2]))
    for name, trace in traces:
        here = frame.to_grid(trace, rotation, center)
        within = (
            inside(outline, here[:, :2])
            & (here[:, 2] > local[:, 2].min() - spread)
            & (here[:, 2] < local[:, 2].max() + spread)
        )
        if within.mean() > 0.5:
            print(
                f"  guided by {name}: {int(within.sum())} of its {len(trace)} points run "
                f"inside, within {spread:.1f} m of the band this outline occupies"
            )
            data = np.vstack([data, here[within]])
    mesh, triangles = mesh_polygon(outline, step, data[:, :2])

    # the analytic surface: the smoothest w(u, v) through the survey's points
    surface = RBFInterpolator(data[:, :2], data[:, 2], kernel="thin_plate_spline")
    vertices = frame.to_dtm(np.column_stack([mesh, surface(mesh)]), rotation, center)
    at_data = np.abs(surface(data[:, :2]) - data[:, 2]).max()
    print(f"  thin-plate spline through those {len(data)} points, off them by at most {at_data:.2e} m")

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


def main():
    surveys = [(path, load(path)) for path in paths()]
    traces = [(path.name, points) for path, points in surveys if not is_closed(points)]
    for name, _ in traces:
        print(f"{name}: open, kept as a guide rather than triangulated")
    for path, points in surveys:
        if is_closed(points):
            print(f"{path.name}:")
            surface_of(path, points, traces)


if __name__ == "__main__":
    main()
