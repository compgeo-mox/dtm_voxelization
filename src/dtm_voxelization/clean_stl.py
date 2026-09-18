"""Tidy up a reconstructed surface: drop the patches that float free of it,
triangulate over the holes it has left, and smooth it lightly.

`python -m dtm_voxelization.clean_stl IN.stl OUT.stl` does it to a surface that
already exists -- las_export runs the same on the STL it writes.

The order matters: patches first, so no hole is closed against a piece of
rubbish, and smoothing last, over the filled mesh.

Holes are filled by VTK. Open3D's own fill_holes was tried first and is not
usable here: it saturates (5, 15 and 30 m thresholds all added the same 1454
triangles on the San Martino crop) and it leaves a mesh that makes the
smoothing diverge -- vertices moved 1e11 m, non-manifold edges up from 88 to
107, and repairing those did not help.

Smoothing is Taubin's, which alternates a shrinking and an expanding pass and
so keeps the surface where it is, unlike plain Laplacian smoothing. It runs
over the whole mesh, since "only the noisy parts" would need a criterion for
noise, but at these few iterations it barely moves a clean area: the log gives
the average and the largest distance travelled, and a move beyond 1% of the
mesh's own diagonal fails the run rather than writing a surface turned inside
out.
"""

import sys
import time
from pathlib import Path

import meshio
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter

from .terrain import export_triangles

MIN_PATCH_AREA = 0.01  # of the largest connected patch's area
HOLE_SIZE = 20.0  # m, the largest hole RADIUS to fill: keep it well under the
# radius of the surface's own outer rim, which is a hole like any other to VTK
SMOOTHING_ITERATIONS = 5
MAX_SMOOTHING_MOVE = 0.01  # of the mesh diagonal, before the run is a failure


def to_polydata(vertices, triangles):
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=float), deep=True))
    polys = vtkCellArray()
    polys.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(triangles) + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(triangles.ravel(), dtype=np.int64), deep=True),
    )
    mesh = vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(polys)
    return mesh


def from_polydata(mesh):
    polys = mesh.GetPolys()
    sizes = np.diff(vtk_to_numpy(polys.GetOffsetsArray()))
    if len(sizes) and sizes.max() != 3:
        raise RuntimeError(f"expected triangles, got cells of up to {sizes.max()} points")
    return (
        vtk_to_numpy(mesh.GetPoints().GetData()),
        vtk_to_numpy(polys.GetConnectivityArray()).reshape(-1, 3),
    )


def boundary_edges(triangles):
    """Edges with only one triangle on them: the rim of the surface and of
    every hole in it."""
    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def clean(vertices, triangles):
    """(vertices, triangles), detached patches dropped, holes filled, smoothed."""
    import open3d as o3d  # as in terrain.py: the heavy import only where used

    t0 = time.perf_counter()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    labels, _, areas = mesh.cluster_connected_triangles()
    labels, areas = np.asarray(labels), np.asarray(areas)
    small = areas < MIN_PATCH_AREA * areas.max()
    print(
        f"  {len(areas):,} connected patches, largest {areas.max():,.0f} m2: dropping "
        f"{int(small[labels].sum()):,} triangles in the {int(small.sum()):,} under "
        f"{100 * MIN_PATCH_AREA:g}% of it",
        flush=True,
    )
    mesh.remove_triangles_by_mask(small[labels])
    mesh.remove_unreferenced_vertices()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    before = boundary_edges(triangles)
    filler = vtkFillHolesFilter()
    filler.SetInputData(to_polydata(vertices, triangles))
    filler.SetHoleSize(HOLE_SIZE)
    normals = vtkPolyDataNormals()  # the new triangles come out wound any which way
    normals.SetInputConnection(filler.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    filled, filled_triangles = from_polydata(normals.GetOutput())
    after = boundary_edges(filled_triangles)
    print(
        f"  holes up to {HOLE_SIZE} m in radius: {len(filled_triangles) - len(triangles):,} "
        f"triangles added, boundary edges {before:,} -> {after:,}",
        flush=True,
    )
    if after == 0:
        print(
            "  WARNING: no boundary left -- HOLE_SIZE is wide enough to have capped the "
            "surface's own outer rim",
            flush=True,
        )

    smoothed = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(filled), o3d.utility.Vector3iVector(filled_triangles)
    ).filter_smooth_taubin(number_of_iterations=SMOOTHING_ITERATIONS)
    moved = np.linalg.norm(np.asarray(smoothed.vertices) - filled, axis=1)
    diagonal = np.linalg.norm(np.ptp(filled, axis=0))
    print(
        f"  {SMOOTHING_ITERATIONS} Taubin iterations: vertices moved {moved.mean():.3f} m on "
        f"average, {np.percentile(moved, 99):.3f} m at the 99th percentile, {moved.max():.3f} m "
        f"at most; {time.perf_counter() - t0:.1f}s in total",
        flush=True,
    )
    if moved.max() > MAX_SMOOTHING_MOVE * diagonal:
        raise RuntimeError(
            f"smoothing moved a vertex by {moved.max():.3f} m, more than "
            f"{100 * MAX_SMOOTHING_MOVE:g}% of the mesh's {diagonal:.1f} m diagonal: the filter "
            f"has diverged, which it does on a mesh whose holes were filled badly"
        )
    return np.asarray(smoothed.vertices), np.asarray(smoothed.triangles)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    mesh = meshio.read(argv[0])
    triangles = [block.data for block in mesh.cells if block.type == "triangle"]
    if not triangles:
        raise SystemExit(f"{argv[0]}: no triangles")
    triangles = np.vstack(triangles)
    print(f"read {argv[0]}: {len(mesh.points):,} points, {len(triangles):,} triangles", flush=True)
    export_triangles(Path(argv[1]), *clean(mesh.points, triangles))


if __name__ == "__main__":
    main()
