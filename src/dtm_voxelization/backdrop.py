"""The land and lake around the case, a backdrop for the renders:

    python -m dtm_voxelization.backdrop [CASE.toml]

A square EXTENT metres a side around the DTM's rectangle, with a hole where
the DTM is: the case's own textured terrain (orthophoto) goes in it. Its
ground is the Copernicus DEM GLO-30 (ESA, 30 m, open data on AWS, read
straight from its cloud-optimised GeoTIFFs), a surface model -- trees and
roofs on it -- which does at that scale. It is sampled every STEP metres on a
grid aligned with the DTM's rectangle, so the hole's edges are grid lines;
on them the height is the DTM's, and outside it the DEM's, shifted by the
DTM-minus-DEM of the nearest dry edge point, a shift that fades over BLEND
metres: the backdrop meets the case's terrain without a step.

Where the DEM is at the lake's level the ground is the lake: it is lowered
under it, and the lake's surface, flat at orthophoto.LAKE_LEVEL, goes over
it, with the water texture and ripples of orthophoto.

The photo is the same national orthophoto as pcn_orthophoto's, fetched over
the square at PIXEL metres and graded as orthophoto grades it. What is
fetched is kept in the repository, so it is fetched once:
data/orthophoto/<case>_backdrop_pcn2012.tif, the photo, and
data/dem/<case>_backdrop_copernicus.tif, the DEM on the grid. Writes, in the
case's output folder,

  backdrop.png        the texture, north up
  backdrop.vtp        the ground around the hole, with its texture coordinates
  backdrop_lake.vtp   the lake around it, texture coordinates in wave tiles
"""

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from scipy.spatial import cKDTree

if __package__:
    from .case import load_case
    from .dtm_io import _load_xyz, load_dtm_interpolator
    from .orthophoto import LAKE_LEVEL, MAX_TEXTURE, WAVE_TILE, grade, write_vtp
    from .pcn_orthophoto import MAX_SIDE, tile, write_photo
    from .shift_fracture import M
    from .terrain import grid_triangles
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.case import load_case
    from dtm_voxelization.dtm_io import _load_xyz, load_dtm_interpolator
    from dtm_voxelization.orthophoto import LAKE_LEVEL, MAX_TEXTURE, WAVE_TILE, grade, write_vtp
    from dtm_voxelization.pcn_orthophoto import MAX_SIDE, tile, write_photo
    from dtm_voxelization.shift_fracture import M
    from dtm_voxelization.terrain import grid_triangles

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button
EXTENT = 6000.0  # m, the square's side
STEP = 5.0  # m between the backdrop's vertices
PIXEL = 1.0  # m, the backdrop photo's resolution
BLEND = 150.0  # m over which the DTM's offset from the DEM fades
WATER = 2.0  # m over the lake's level under which the DEM is lake
DEM = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
       "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")


def dem(x, y, cache):
    """The Copernicus DEM on the grid x (nx,), y (ny,), DTM frame, as (nx, ny),
    from cache when it is there and on this grid, else fetched into it."""
    west, east, south, north = x[0] + M[0], x[-1] + M[0], y[0] + M[1], y[-1] + M[1]
    transform = rasterio.transform.from_origin(west - STEP / 2, north + STEP / 2, STEP, STEP)
    if cache.exists():
        with rasterio.open(cache) as kept:
            if (kept.width, kept.height) == (len(x), len(y)) and kept.transform == transform:
                print(f"  DEM from {cache}", flush=True)
                return kept.read(1)[::-1].T - M[2]
        print(f"  {cache} is on another grid: fetched again", flush=True)
    lon0, lat0, lon1, lat1 = transform_bounds("EPSG:32632", "EPSG:4326", west, south, east, north)
    heights = np.full((len(y), len(x)), np.nan, dtype=np.float32)  # rows north to south
    for lat in range(int(np.floor(lat0)), int(np.floor(lat1)) + 1):
        for lon in range(int(np.floor(lon0)), int(np.floor(lon1)) + 1):
            url = DEM.format(lat=lat, lon=lon)
            t0 = time.perf_counter()
            part = np.full_like(heights, np.nan)
            with rasterio.open(url) as source:
                reproject(rasterio.band(source, 1), part, dst_transform=transform,
                          dst_crs="EPSG:32632", dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
            heights = np.where(np.isnan(heights), part, heights)
            print(f"  DEM tile N{lat} E{lon:03d}: {100 * np.isfinite(part).mean():.1f} per cent "
                  f"of the grid, in {time.perf_counter() - t0:.1f}s", flush=True)
    if np.isnan(heights).any():
        raise SystemExit(f"{int(np.isnan(heights).sum())} grid points with no DEM")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cache, "w", driver="GTiff", width=len(x), height=len(y), count=1,
                       dtype="float32", crs="EPSG:32632", transform=transform,
                       compress="deflate", predictor=3, tiled=True) as out:
        out.write(heights, 1)
    print(f"  wrote {cache}: {cache.stat().st_size / 1024**2:.1f} MB", flush=True)
    return heights[::-1].T - M[2]  # (nx, ny), y increasing, DTM frame


def photo(west, south, width, height, cache):
    """The national orthophoto over the square, (height, width, 3), north up,
    from cache when it is there, else fetched into it."""
    if cache.exists():
        with rasterio.open(cache) as kept:
            if (kept.width, kept.height) == (width, height):
                print(f"  photo from {cache}", flush=True)
                return np.moveaxis(kept.read(), 0, -1)
        print(f"  {cache} is of another size: fetched again", flush=True)
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    for col in range(0, width, MAX_SIDE):
        for row in range(0, height, MAX_SIDE):
            w, h = min(MAX_SIDE, width - col), min(MAX_SIDE, height - row)
            mosaic[height - row - h: height - row, col: col + w] = tile(
                west + col * PIXEL, south + row * PIXEL, w, h, PIXEL)
        print(f"  photo: column {col} of {width} fetched", flush=True)
    write_photo(cache, mosaic, west, south + height * PIXEL, PIXEL)
    return mosaic


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)

    px, py, _ = _load_xyz(case.points)
    lo, hi = np.array([px.min(), py.min()]), np.array([px.max(), py.max()])
    centre = (lo + hi) / 2
    # the grid's lines run through the rectangle's corner lo, every STEP
    below = np.ceil((lo - (centre - EXTENT / 2)) / STEP)
    above = np.ceil(((centre + EXTENT / 2) - lo) / STEP)
    x = lo[0] + STEP * np.arange(-below[0], above[0] + 1)
    y = lo[1] + STEP * np.arange(-below[1], above[1] + 1)
    print(f"{case.name}: backdrop {x[-1] - x[0]:.0f} x {y[-1] - y[0]:.0f} m every {STEP:g} m "
          f"({len(x):,} x {len(y):,}), around the DTM's {hi[0] - lo[0]:.1f} x "
          f"{hi[1] - lo[1]:.1f} m", flush=True)

    z = dem(x, y, root / "data" / "dem" / f"{case.name}_backdrop_copernicus.tif")
    gx, gy = np.meshgrid(x, y, indexing="ij")  # i * ny + j, as grid_triangles
    tol = 1e-6 * EXTENT
    inside = ((gx >= lo[0] - tol) & (gx <= hi[0] + tol) & (gy >= lo[1] - tol) & (gy <= hi[1] + tol))
    terrain = load_dtm_interpolator(case.points)
    z_dtm = terrain(np.column_stack([gx[inside], gy[inside]]))
    level = LAKE_LEVEL - M[2]
    # the offset is read on the rectangle's edge, and on dry land only: under
    # the lake the DTM is its bed and the DEM its surface
    gap = np.minimum.reduce([gx[inside] - lo[0], hi[0] - gx[inside],
                             gy[inside] - lo[1], hi[1] - gy[inside]])
    edge = (gap < STEP / 2) & (z_dtm > level + WATER)
    offset = z_dtm[edge] - z[inside][edge]
    print(f"  DTM - DEM on {int(edge.sum()):,} dry points of the DTM's edge: median "
          f"{np.median(offset):.1f} m, {np.percentile(offset, 5):.1f} .. "
          f"{np.percentile(offset, 95):.1f} between the 5th and 95th percentiles", flush=True)
    edge_xy = np.column_stack([gx[inside][edge], gy[inside][edge]])
    distance, nearest = cKDTree(edge_xy).query(np.column_stack([gx[~inside], gy[~inside]]))
    z[~inside] += offset[nearest] * np.exp(-distance / BLEND)
    z[inside] = z_dtm

    wet = ~inside & (z < level + WATER)
    z[wet] = np.minimum(z[wet], level - 3.0)  # under the lake's surface, not at it
    print(f"  lake: {100 * wet.mean():.1f} per cent of the backdrop under "
          f"{LAKE_LEVEL + WATER:g} m a.s.l.", flush=True)

    points = np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])
    triangles = grid_triangles(len(x), len(y))
    middle = points[triangles, :2].mean(axis=1)
    hole = ((middle[:, 0] > lo[0]) & (middle[:, 0] < hi[0]) &
            (middle[:, 1] > lo[1]) & (middle[:, 1] < hi[1]))
    west, south = x[0] + M[0], y[0] + M[1]
    width = int(round((x[-1] - x[0]) / PIXEL))
    height = int(round((y[-1] - y[0]) / PIXEL))
    if max(width, height) > MAX_TEXTURE:
        raise SystemExit(f"{width} x {height} pixels: over {MAX_TEXTURE} a side")
    uv = np.column_stack([(points[:, 0] - x[0]) / (x[-1] - x[0]),
                          (points[:, 1] - y[0]) / (y[-1] - y[0])])
    case.output.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    texture = grade(photo(west, south, width, height,
                          root / "data" / "orthophoto" / f"{case.name}_backdrop_pcn2012.tif"))
    png = case.output / "backdrop.png"
    Image.fromarray(texture).save(png)
    print(f"wrote {png}: {width:,} x {height:,} at {PIXEL:g} m, in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    vtp = case.output / "backdrop.vtp"
    write_vtp(vtp, points, triangles[~hole], uv)
    print(f"wrote {vtp}: {int((~hole).sum()):,} triangles around a hole of "
          f"{int(hole.sum()):,}", flush=True)

    wet_triangles = triangles[~hole & wet.ravel()[triangles].any(axis=1)]
    used, compact = np.unique(wet_triangles, return_inverse=True)
    water = points[used].copy()
    water[:, 2] = level
    lake = case.output / "backdrop_lake.vtp"
    write_vtp(lake, water, compact.reshape(-1, 3), water[:, :2] / WAVE_TILE, up=True)
    print(f"wrote {lake}: {len(wet_triangles):,} triangles", flush=True)


if __name__ == "__main__":
    main()
