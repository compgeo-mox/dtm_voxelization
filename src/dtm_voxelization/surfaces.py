"""Fracture surfaces of a case, as triangles in the DTM's frame.

A surface is either an STL plus the translation that brings it into the
DTM's frame, or the corners of a polygon given directly in that frame,
fan-triangulated from the first corner (exact for convex polygons). The grid
lives in its own frame (frame.py), so surfaces are rotated into it before
being written or cut.
"""

import meshio
import numpy as np

from . import frame
from .mesh import read_hex_mesh, require


def triangles(surface):
    """(T, 3, 3) triangle vertices of a surface, in the DTM's frame."""
    if surface.stl is None:
        n = len(surface.corners)
        fan = np.array([(0, k, k + 1) for k in range(1, n - 1)])
        return surface.corners[fan]

    mesh = meshio.read(str(surface.stl))
    tri = [b.data for b in mesh.cells if b.type == "triangle"]
    if not tri:
        raise ValueError(f"{surface.stl}: no triangles")
    points = mesh.points.astype(float) + np.asarray(surface.translation, dtype=float)
    return points[np.vstack(tri)]


def in_grid_frame(case, surface):
    """Triangles of a surface in the grid's frame."""
    require(case.frame_path, "grid")
    rotation, center = frame.load(case.frame_path)
    tri = triangles(surface)
    return frame.to_grid(tri.reshape(-1, 3), rotation, center).reshape(tri.shape)


def run(case):
    """Write each surface, in the grid's frame, to `surfaces/<name>.stl`."""
    if not case.surfaces:
        print("  no surfaces in the case", flush=True)
        return

    grid_bbox = None
    if case.grid_path.exists():
        points, _ = read_hex_mesh(case.grid_path)
        grid_bbox = (points.min(axis=0), points.max(axis=0))

    case.surface_dir.mkdir(parents=True, exist_ok=True)
    for surface in case.surfaces:
        tri = in_grid_frame(case, surface)
        lo, hi = tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)
        source = surface.stl if surface.stl is not None else f"{len(surface.corners)} corners"
        print(f"  {surface.name}: {len(tri)} triangles from {source}", flush=True)
        print(f"    translation {surface.translation}, bbox {lo.round(3)} .. {hi.round(3)}")
        if grid_bbox is not None:
            overlaps = bool(np.all((hi >= grid_bbox[0]) & (lo <= grid_bbox[1])))
            print(f"    overlaps the grid's bounding box: {overlaps}", flush=True)

        out_path = case.surface_dir / f"{surface.name}.stl"
        meshio.write_points_cells(
            out_path, tri.reshape(-1, 3), [("triangle", np.arange(3 * len(tri)).reshape(-1, 3))]
        )
        print(f"    wrote {out_path}", flush=True)
