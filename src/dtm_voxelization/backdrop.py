"""The land and lake around the case, a backdrop for the renders:

    python -m dtm_voxelization.backdrop [CASE.toml]

A square EXTENT metres a side around the DTM's rectangle, with a hole where
the DTM is: the case's own textured terrain (orthophoto) goes in it. Its
ground is Regione Lombardia's DTM 5x5 (2015, bare ground, from its
geoportal's ImageServer). It is sampled every STEP metres on a
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
data/dem/<case>_backdrop_dtm5.tif, the DEM on the grid. Writes, in the
case's output folder,

  backdrop.png        the texture, north up
  backdrop.vtp        the ground around the hole, with its texture coordinates
  backdrop_lake.vtp   the lake around it, texture coordinates in wave tiles
"""

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.spatial import cKDTree

if __package__:
    from .case import load_case
    from .dtm_io import _load_xyz, load_dtm_interpolator
    from .orthophoto import LAKE_LEVEL, MAX_TEXTURE, WAVE_TILE, grade, write_vtp
    from .pcn_orthophoto import tile, write_photo
    from .shift_fracture import M
    from .terrain import grid_triangles
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.case import load_case
    from dtm_voxelization.dtm_io import _load_xyz, load_dtm_interpolator
    from dtm_voxelization.orthophoto import LAKE_LEVEL, MAX_TEXTURE, WAVE_TILE, grade, write_vtp
    from dtm_voxelization.pcn_orthophoto import tile, write_photo
    from dtm_voxelization.shift_fracture import M
    from dtm_voxelization.terrain import grid_triangles

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button
EXTENT = 6000.0  # m, the square's side
STEP = 5.0  # m between the backdrop's vertices
PIXEL = 1.0  # m, the backdrop photo's resolution
BLEND = 150.0  # m over which the DTM's offset from the DEM fades
WATER = 2.0  # m over the lake's level under which the DEM is lake
SIDE = 500  # pixels a side per photo request: 2 km ones hit a broken file of the service
DEM = "https://www.cartografia.servizirl.it/arcgis2/rest/services/BaseMap/DTM5_RL_img/ImageServer"


def dem(x, y, cache):
    """Regione Lombardia's DTM 5x5 on the grid x (nx,), y (ny,), DTM frame, as
    (nx, ny), from cache when it is there and on this grid, else fetched into it."""
    west, east, south, north = x[0] + M[0], x[-1] + M[0], y[0] + M[1], y[-1] + M[1]
    transform = rasterio.transform.from_origin(west - STEP / 2, north + STEP / 2, STEP, STEP)
    if cache.exists():
        with rasterio.open(cache) as kept:
            if (kept.width, kept.height) == (len(x), len(y)) and kept.transform == transform:
                print(f"  DEM from {cache}", flush=True)
                return kept.read(1)[::-1].T - M[2]
        print(f"  {cache} is on another grid: fetched again", flush=True)
    query = urllib.parse.urlencode({
        "bbox": f"{west - STEP / 2},{south - STEP / 2},{east + STEP / 2},{north + STEP / 2}",
        "bboxSR": 32632, "imageSR": 32632, "size": f"{len(x)},{len(y)}", "format": "tiff",
        "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "f": "image"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(f"{DEM}/exportImage?{query}", timeout=300) as response:
        body = response.read()
    with rasterio.MemoryFile(body) as memory, memory.open() as source:
        heights = source.read(1).astype(np.float32)  # rows north to south
        if (source.width, source.height) != (len(x), len(y)):
            raise SystemExit(f"asked {len(x)} x {len(y)} DEM pixels, got {source.width} x {source.height}")
        if source.nodata is not None:
            heights[heights == source.nodata] = np.nan
    print(f"  DEM {len(x):,} x {len(y):,} from {DEM}, {len(body) / 1024**2:.1f} MB in "
          f"{time.perf_counter() - t0:.1f}s: {np.nanmin(heights):.1f} .. {np.nanmax(heights):.1f} m",
          flush=True)
    if np.isnan(heights).any() or np.nanmin(heights) < 100:
        raise SystemExit(f"{int(np.isnan(heights).sum())} grid points with no DEM, "
                         f"lowest {np.nanmin(heights):.1f} m")
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
    missing = []

    def fetch(col, row, w, h):
        """A piece, split in four when the service fails on it, down to SIDE/4;
        left black when it still fails."""
        try:
            mosaic[height - row - h: height - row, col: col + w] = tile(
                west + col * PIXEL, south + row * PIXEL, w, h, PIXEL)
        except (OSError, SystemExit) as error:
            if max(w, h) <= SIDE // 4:
                print(f"  PHOTO MISSING at column {col}, row {row} from the south, {w} x {h}: "
                      f"left black ({error})", flush=True)
                missing.append(w * h)
                return
            print(f"  failed at column {col}, row {row}, {w} x {h}: split in four", flush=True)
            hw, hh = (w + 1) // 2, (h + 1) // 2
            for c, r, ww, rh in ((col, row, hw, hh), (col + hw, row, w - hw, hh),
                                 (col, row + hh, hw, h - hh), (col + hw, row + hh, w - hw, h - hh)):
                if ww and rh:
                    fetch(c, r, ww, rh)

    for col in range(0, width, SIDE):
        for row in range(0, height, SIDE):
            fetch(col, row, min(SIDE, width - col), min(SIDE, height - row))
        print(f"  photo: column {col} of {width} fetched", flush=True)
    if missing:
        print(f"  PHOTO MISSING on {len(missing)} pieces, {100 * sum(missing) / (width * height):.2f} "
              f"per cent of the square, left black", flush=True)
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

    z = dem(x, y, root / "data" / "dem" / f"{case.name}_backdrop_dtm5.tif")
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
