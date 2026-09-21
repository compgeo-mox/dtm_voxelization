"""A LAS point cloud as the repo's .xyz, plus a .vtp and an .stl for ParaView:
`python -m dtm_voxelization.las_export CLOUD.las CASE.toml`, all three written
next to the cloud.

All three are in the frame the cases use (see the README's Data section):
x/y centred on the cloud's mean, z elevation. The mean is rounded to 1 mm, the
LAS resolution, so the centred coordinates stay exact at 1 mm; it is logged and
written in the .xyz comment line, to add it back for UTM.

- .xyz: what a case's `points` reads.
- .vtp: every point, binary. A text cloud this size is no good in ParaView,
  whose CSV reader parses each field as a string and runs out of memory.
- .stl: the surface a `poisson` case reconstructs from these points, with the
  case's `outward`, `voxel_size` and `poisson_depth`, tidied up by clean_stl.
  The reconstruction is closed, and away from the data it closes itself with
  surfaces of no meaning, so only the triangles within MAX_GAP of a point are
  kept -- which is the same thing as deciding how wide a gap in the cloud the
  reconstruction is allowed to bridge on its own. Every hole in the exported
  surface is one this trim opened.
"""

import sys
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

from . import frame
from .case import load_case
from .dtm_io import save_xyz
from .clean_stl import clean
from .terrain import compact_triangles, export_triangles, poisson_surface

MAX_GAP = 5.0  # m; the surface is kept this far from the cloud, so the
# reconstruction bridges anything narrower and wider voids stay open


def write_vtp(path, points):
    """Every point, in one poly-vertex cell so ParaView shows them as they are."""
    cloud_points = vtkPoints()
    cloud_points.SetData(numpy_to_vtk(points, deep=False))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.array([0, len(points)], dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(len(points), dtype=np.int64), deep=True),
    )
    cloud = vtkPolyData()
    cloud.SetPoints(cloud_points)
    cloud.SetVerts(cells)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(cloud)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToLZ4()
    if not writer.Write():
        raise RuntimeError(f"vtkXMLPolyDataWriter failed on {path}")
    print(f"wrote {path}: {path.stat().st_size / 1024**2:.0f} MB", flush=True)


def write_poisson_stl(path, points, case):
    """The case's Poisson surface of the points, where there are points."""
    surface, rotation, center = poisson_surface(points, case)
    t0 = time.perf_counter()
    centroids = surface.vertices[surface.triangles].mean(axis=1)
    distance, _ = cKDTree(surface.points).query(
        centroids, distance_upper_bound=MAX_GAP, workers=-1
    )
    near = np.isfinite(distance)
    print(
        f"  {int(near.sum()):,} of {len(near):,} triangles within {MAX_GAP} m of a point, "
        f"found in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    vertices, triangles, _ = compact_triangles(
        frame.to_dtm(surface.vertices, rotation, center), surface.triangles, near
    )
    export_triangles(path, *clean(vertices, triangles))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    las_path = Path(argv[0]).resolve()
    case = load_case(argv[1])
    if case.surface != "poisson":
        raise SystemExit(f"{argv[1]}: surface is {case.surface!r}, the .stl needs a 'poisson' case")
    print(
        f"case {case.name}: outward {case.outward}, voxel {case.voxel_size}, "
        f"depth {case.poisson_depth}, trim_to_footprint={case.trim_to_footprint}",
        flush=True,
    )

    t0 = time.perf_counter()
    las = laspy.read(las_path)
    points = np.column_stack([las.x, las.y, las.z])
    print(f"read {las_path}: {len(points):,} points in {time.perf_counter() - t0:.1f}s", flush=True)

    mean = np.round(points[:, :2].mean(axis=0), 3)
    points[:, :2] -= mean
    print(
        f"  x/y centred on {mean[0]:.3f} {mean[1]:.3f} (add back for UTM): "
        f"x [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] "
        f"y [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] "
        f"z [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]",
        flush=True,
    )

    stem = las_path.with_suffix("")
    save_xyz(
        stem.with_suffix(".xyz"),
        points,
        f"{las_path.name}, x/y centred on {mean[0]:.3f} {mean[1]:.3f}, z elevation",
    )
    write_vtp(stem.with_suffix(".vtp"), points)
    write_poisson_stl(stem.with_suffix(".stl"), points, case)


if __name__ == "__main__":
    main()
