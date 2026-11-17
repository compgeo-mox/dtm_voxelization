"""Orthophotos as a texture over the terrain, to look at in ParaView:

    python -m dtm_voxelization.orthophoto [CASE.toml PIXEL PHOTO [PHOTO ...]]

The photos, GeoTIFFs in UTM 32N, are resampled at PIXEL metres onto the DTM's
rectangle and laid one over the other, each later one over the earlier ones
wherever it has a picture -- at Rialba the national orthophoto of
pcn_orthophoto alone, which covers the whole mountain. The
result is one texture, and the terrain a mesh every MESH_STEP metres carrying
its texture coordinates, so the photo keeps its own detail however coarse the
mesh -- the DTM has a 5 m step, so a finer mesh would add nothing.

PIXEL sets the texture's detail and weight: 0.5 m (3.5k x 3.2k pixels at
Rialba) is the national photo's own; finer only pays with a finer photo on
top. A texture can be at most MAX_TEXTURE pixels a side, as GPUs take it.

Writes, in the case's output folder,

  orthophoto_<PIXEL>m.png   the texture, north up
  orthophoto_<PIXEL>m.vtp   the terrain with its texture coordinates

In ParaView: open the .vtp, Coloring Solid Color (white), and under
Properties > Texture load the .png.

With no arguments it takes CASE and the PCN mosaic alone at its own PIXEL, so the file runs from an editor as it does from the shell. The
photo's own georeferencing is used, not a .tfw beside it: at Rialba the
drone's disagrees by 38 m in x.
"""

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject

if __package__:
    from . import terrain as terrain_surfaces
    from .case import load_case
    from .dtm_io import _load_xyz, interpolate_in_parallel
    from .shift_fracture import M  # the same Rialba mean
    from .terrain import grid_triangles
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization import terrain as terrain_surfaces
    from dtm_voxelization.case import load_case
    from dtm_voxelization.dtm_io import _load_xyz, interpolate_in_parallel
    from dtm_voxelization.shift_fracture import M
    from dtm_voxelization.terrain import grid_triangles

MESH_STEP = 2.0  # m between the terrain mesh's vertices
MAX_TEXTURE = 16384  # pixels a side
# what to work on when no argument is given, as from an editor's Run button;
# both photos are resolved against the repository
CASE = "cases/rialba.toml"
PIXEL = 0.5
PHOTOS = ["output/rialba/pcn2012.tif"]  # one photo over the whole mountain


def canvas(case, pixel):
    """(west, north, width, height) of the texture over the DTM's rectangle, UTM."""
    x, y, _ = _load_xyz(case.points)
    west, south = x.min() + M[0], y.min() + M[1]
    width = int(np.ceil((x.max() + M[0] - west) / pixel))
    height = int(np.ceil((y.max() + M[1] - south) / pixel))
    if max(width, height) > MAX_TEXTURE:
        raise SystemExit(f"{width} x {height} pixels at {pixel} m: over {MAX_TEXTURE} a side, "
                         f"take a PIXEL of at least "
                         f"{max(width, height) * pixel / MAX_TEXTURE:.3f} m")
    return west, south + height * pixel, width, height


def paint(texture, path, transform):
    """The photo at path laid over texture (h, w, 3) where it has a picture."""
    t0 = time.perf_counter()
    height, width = texture.shape[:2]
    with rasterio.open(path) as photo:
        finer = photo.res[0] < abs(transform.a)
        resampling = Resampling.average if finer else Resampling.bilinear
        bands = np.zeros((3, height, width), dtype=np.uint8)
        for b in range(3):
            reproject(rasterio.band(photo, b + 1), bands[b], dst_transform=transform,
                      dst_crs="EPSG:32632", resampling=resampling)
        # where the photo has a picture: its alpha band, or its nodata mask
        seen = np.zeros((height, width), dtype=np.uint8)
        mask = photo.read_masks(1) if photo.count < 4 else photo.read(4)
        reproject(mask, seen, src_transform=photo.transform, src_crs=photo.crs,
                  dst_transform=transform, dst_crs="EPSG:32632",
                  resampling=Resampling.nearest)
        print(f"  {path}: {photo.width:,} x {photo.height:,} at {photo.res[0]:.3f} m, "
              f"{photo.crs}, {resampling.name} -> covers {100 * (seen > 0).mean():.1f} "
              f"per cent of the texture, in {time.perf_counter() - t0:.1f}s", flush=True)
    covered = seen > 0
    texture[covered] = np.moveaxis(bands, 0, -1)[covered]
    return covered


def terrain_mesh(terrain, west, north, width, height, pixel):
    """(points, triangles, uv) of the terrain over the texture's rectangle, in
    the DTM's frame, uv = (0, 0) at the texture's south-west corner."""
    east = np.arange(west, west + width * pixel + MESH_STEP / 2, MESH_STEP)
    south = north - height * pixel
    northing = np.arange(south, north + MESH_STEP / 2, MESH_STEP)
    x, y = np.meshgrid(east - M[0], northing - M[1], indexing="ij")  # i * ny + j
    t0 = time.perf_counter()
    z = interpolate_in_parallel(terrain.interp, np.column_stack([x.ravel(), y.ravel()]))
    print(f"  terrain at {x.size:,} points every {MESH_STEP:g} m in "
          f"{time.perf_counter() - t0:.1f}s, z {np.nanmin(z):.1f} .. {np.nanmax(z):.1f}",
          flush=True)
    if np.isnan(z).any():
        raise SystemExit(f"{int(np.isnan(z).sum())} mesh points off the DTM")
    points = np.column_stack([x.ravel(), y.ravel(), z])
    uv = np.column_stack([(x.ravel() + M[0] - west) / (width * pixel),
                          (y.ravel() + M[1] - south) / (height * pixel)])
    return points, grid_triangles(len(east), len(northing)), uv


def write_vtp(path, points, triangles, uv):
    """The surface as VTK PolyData, uv as its active texture coordinates."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    surface = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    surface.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(triangles) + 1, 3), deep=True),
                  numpy_to_vtkIdTypeArray(triangles.ravel().astype(np.int64), deep=True))
    surface.SetPolys(cells)
    tcoords = numpy_to_vtk(np.ascontiguousarray(uv, dtype=np.float32), deep=True)
    tcoords.SetName("uv")
    surface.GetPointData().SetTCoords(tcoords)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    if not writer.Write():
        raise SystemExit(f"could not write {path}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1 or len(argv) == 2:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)
    pixel = float(argv[1]) if argv else PIXEL
    photos = [Path(p).resolve() for p in argv[2:]] if argv else [root / p for p in PHOTOS]
    if case.surface != "height_field":
        raise SystemExit(f"{case.name}: surface is {case.surface!r}, draping needs a height field")

    west, north, width, height = canvas(case, pixel)
    print(f"{case.name}: texture {width:,} x {height:,} pixels at {pixel:g} m, from "
          f"({west:.1f}, {north:.1f}) UTM 32N; {len(photos)} photo(s), the last on top",
          flush=True)
    transform = rasterio.transform.from_origin(west, north, pixel, pixel)
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    covered = np.zeros((height, width), dtype=bool)
    for photo in photos:
        covered |= paint(texture, photo, transform)
    print(f"  {100 * (1 - covered.mean()):.1f} per cent of the texture with no photo (black)",
          flush=True)

    case.output.mkdir(parents=True, exist_ok=True)
    png = case.output / f"orthophoto_{pixel:g}m.png"
    vtp = case.output / f"orthophoto_{pixel:g}m.vtp"
    t0 = time.perf_counter()
    Image.fromarray(texture).save(png)
    print(f"wrote {png}: {png.stat().st_size / 1024**2:.1f} MB in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    terrain, _, _ = terrain_surfaces.build(case)
    points, triangles, uv = terrain_mesh(terrain, west, north, width, height, pixel)
    write_vtp(vtp, points, triangles, uv)
    print(f"wrote {vtp}: {len(points):,} points, {len(triangles):,} triangles, "
          f"{vtp.stat().st_size / 1024**2:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
