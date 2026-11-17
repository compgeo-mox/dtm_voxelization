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
  lake_<PIXEL>m.vtp         the lake's surface, flat at LAKE_LEVEL, its
                            texture coordinates in tiles of WAVE_TILE metres
  lake_water.png            a tile of water, in the lake's mean colour in the photo
  lake_normal.png           the same tile's waves as a normal map

The DTM carries the lake's bathymetry, so the photo of the water lies on the
lake's bed; the lake's surface goes over it, wherever the terrain dips under
LAKE_LEVEL, and meets the shore where the terrain crosses the level, so it
needs no outline of its own. Its waves are a height field of filtered noise,
periodic over the tile, turned into a tangent-space normal map; the surface
carries the normals and tangents a normal map needs.

In ParaView: open the terrain's .vtp, Coloring Solid Color (white), and under
Properties > Texture load orthophoto_<PIXEL>m.png. Open the lake's .vtp,
Coloring Solid Color, Interpolation PBR, Texture lake_water.png, Normal
Texture lake_normal.png, Roughness about 0.1 and Opacity about 0.9.

With no arguments it takes CASE and the PCN mosaic alone at its own PIXEL,
so the file runs from an editor as it does from the shell. The
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
# m above the sea, Lake Como's mean level: where the DTM's shore points stop
# and its bathymetry starts (197.x below, 199 and up above)
LAKE_LEVEL = 198.0
WAVE_TILE = 25.0  # m of lake one tile of waves covers
WAVE_PIXELS = 1024  # a tile's side, in pixels
WATER_GAIN = 1.5  # the photo's water, seen straight down, is darker than from a slant
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


def waves(colour, seed=0):
    """(water, normal) images of one periodic tile of waves: water in the given
    colour, a shade lighter on the crests, and their tangent-space normals."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(WAVE_PIXELS) * WAVE_PIXELS  # cycles per tile
    kx, ky = np.meshgrid(k, k, indexing="xy")
    radius = np.hypot(kx, ky)
    # most of the energy at 8 to 30 waves a tile, i.e. ripples of 1 to 3 m,
    # stretched along x as a breeze would
    spectrum = np.exp(-((radius - 15) / 8) ** 2) * (1 + 0.6 * (np.abs(kx) > np.abs(ky)))
    height = np.real(np.fft.ifft2(spectrum * np.exp(2j * np.pi * rng.random(radius.shape))))
    height /= np.abs(height).max()
    slope = 6.0  # how steep the ripples look
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * slope / 2
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * slope / 2
    normal = np.dstack([-dx, dy, np.ones_like(height)])  # image rows run south: dy flips
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    shade = 1 + 0.12 * height[..., None]
    water = np.clip(np.asarray(colour, float) * shade, 0, 255).astype(np.uint8)
    return water, ((normal * 0.5 + 0.5) * 255).astype(np.uint8)


def write_vtp(path, points, triangles, uv, up=False):
    """The surface as VTK PolyData, uv as its active texture coordinates; with
    up, a flat surface, also +z normals and +x tangents, for a normal map."""
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
    if up:
        normals = numpy_to_vtk(np.tile(np.float32([0, 0, 1]), (len(points), 1)), deep=True)
        normals.SetName("Normals")
        surface.GetPointData().SetNormals(normals)
        tangents = numpy_to_vtk(np.tile(np.float32([1, 0, 0]), (len(points), 1)), deep=True)
        tangents.SetName("Tangents")
        surface.GetPointData().SetTangents(tangents)
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

    level = LAKE_LEVEL - M[2]  # in the DTM's frame
    wet = (points[triangles, 2] < level).any(axis=1)
    print(f"  lake at {LAKE_LEVEL:g} m a.s.l. (z {level:.2f}): {int(wet.sum()):,} of "
          f"{len(triangles):,} triangles have a corner under it", flush=True)
    if wet.any():
        used, compact = np.unique(triangles[wet], return_inverse=True)
        water = points[used].copy()
        water[:, 2] = level
        # the photo's colour of the water: its texture where the lake is deep,
        # away from the shore's rocks and road
        deep = points[used, 2] < level - 5
        cols = np.clip((uv[used][deep, 0] * width).astype(int), 0, width - 1)
        rows = np.clip(((1 - uv[used][deep, 1]) * height).astype(int), 0, height - 1)
        colour = np.clip(WATER_GAIN * np.median(texture[rows, cols], axis=0), 0, 255)
        print(f"  water colour {colour.astype(int).tolist()}, {WATER_GAIN:g} times the "
              f"photo's median over {int(deep.sum()):,} points of deep water", flush=True)
        tile, normal = waves(colour)
        for name, image in (("lake_water.png", tile), ("lake_normal.png", normal)):
            Image.fromarray(image).save(case.output / name)
            print(f"wrote {case.output / name}: {WAVE_PIXELS} pixels for {WAVE_TILE:g} m",
                  flush=True)
        lake = case.output / f"lake_{pixel:g}m.vtp"
        write_vtp(lake, water, compact.reshape(-1, 3), water[:, :2] / WAVE_TILE, up=True)
        print(f"wrote {lake}: {len(water):,} points, "
              f"{(np.ptp(water[:, 0]) * np.ptp(water[:, 1])) / 1e6:.2f} km2 bounding box",
              flush=True)

if __name__ == "__main__":
    main()
