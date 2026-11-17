"""The national orthophoto over the whole case, as a GeoTIFF:

    python -m dtm_voxelization.pcn_orthophoto [CASE.toml]

AGEA's 2012 colour orthophoto (50 cm), served by the Geoportale Nazionale
(PCN) through its WMS, whose capabilities state no access constraints and no
fees; credit it as "Geoportale Nazionale - Ortofoto AGEA 2012". It covers the
whole DTM, mountain, lake and road, where the survey's drone photo covers
the wall alone: orthophoto.py lays the drone's over it.

The WMS serves at most MAX_SIDE pixels a side, so the DTM's rectangle is
fetched in tiles at PIXEL metres, in UTM 32N directly, and mosaicked into
output/<case>/pcn2012.tif.
"""

import io
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

if __package__:
    from .case import load_case
    from .dtm_io import _load_xyz
    from .shift_fracture import M
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.case import load_case
    from dtm_voxelization.dtm_io import _load_xyz
    from dtm_voxelization.shift_fracture import M

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button
WMS = "http://wms.pcn.minambiente.it/ogc?map=/ms_ogc/WMS_v1.3/raster/ortofoto_colore_12.map"
LAYER = "OI.ORTOIMMAGINI.2012.32"  # the UTM 32N half of the country
PIXEL = 0.5  # m, the photo's own resolution
MAX_SIDE = 2000  # pixels a side per request, under the service's 2048


def tile(west, south, width, height):
    """(height, width, 3) of the photo over that rectangle, UTM 32N."""
    query = urllib.parse.urlencode({
        "service": "WMS", "version": "1.3.0", "request": "GetMap", "layers": LAYER,
        "styles": "", "crs": "EPSG:32632", "format": "image/png",
        "bbox": f"{west},{south},{west + width * PIXEL},{south + height * PIXEL}",
        "width": width, "height": height})
    request = urllib.request.Request(f"{WMS}&{query}",
                                     headers={"User-Agent": "dtm_voxelization orthophoto"})
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
            image = Image.open(io.BytesIO(body))
            break
        except (OSError, urllib.error.URLError) as error:
            print(f"    attempt {attempt}: {error}", flush=True)
            if attempt == 5:
                raise
            time.sleep(10 * attempt)
    rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != (height, width):
        raise SystemExit(f"asked {width} x {height} pixels, got {rgb.shape[1]} x {rgb.shape[0]}")
    return rgb


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)

    x, y, _ = _load_xyz(case.points)
    west, south = np.floor(x.min() + M[0]), np.floor(y.min() + M[1])
    width = int(np.ceil((x.max() + M[0] - west) / PIXEL))
    height = int(np.ceil((y.max() + M[1] - south) / PIXEL))
    print(f"{case.name}: {width:,} x {height:,} pixels at {PIXEL:g} m from ({west:.0f}, "
          f"{south:.0f}) UTM 32N, in tiles of at most {MAX_SIDE}", flush=True)

    mosaic = np.zeros((height, width, 3), dtype=np.uint8)  # row 0 is the north edge
    for col in range(0, width, MAX_SIDE):
        for row in range(0, height, MAX_SIDE):
            w, h = min(MAX_SIDE, width - col), min(MAX_SIDE, height - row)
            t0 = time.perf_counter()
            mosaic[height - row - h: height - row, col: col + w] = tile(
                west + col * PIXEL, south + row * PIXEL, w, h)
            print(f"  tile at column {col}, row {row} from the south: {w} x {h} in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    empty = (mosaic.max(axis=2) == 0).mean()
    print(f"  {100 * empty:.1f} per cent of the pixels black (no photo)", flush=True)

    case.output.mkdir(parents=True, exist_ok=True)
    path = case.output / "pcn2012.tif"
    transform = rasterio.transform.from_origin(west, south + height * PIXEL, PIXEL, PIXEL)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3,
                       dtype="uint8", crs="EPSG:32632", transform=transform,
                       compress="deflate", tiled=True) as out:
        out.write(np.moveaxis(mosaic, -1, 0))
    print(f"wrote {path}: {path.stat().st_size / 1024**2:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
