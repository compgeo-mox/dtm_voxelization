"""The orthophoto draped over the terrain, to look at in ParaView, plus a
downsampled copy of the photo itself:

    python -m dtm_voxelization.orthophoto ORTHO.tif CASE.toml

Both are written next to the photo. The colour -- rock, scree, trees -- is the
photo's; the shape is the case's DTM, which at Rialba has a 5 m step and so
stays smooth under it, exactly as a satellite view is a sharp image on a soft
terrain.

The mesh is one vertex per pixel of the DOWNSAMPLED photo, coloured by that
pixel, so TARGET_PIXELS sets both the photo's resolution and the mesh's: a
5 cm photo of 271 Mpixel is far past what a screen or a GPU can use, and
draping it whole would cost a mesh of the same size. Pixels the photo marks
as transparent carry no colour, so triangles touching one are dropped rather
than filled with whatever lies under them.

The photo is georeferenced in UTM32N and the case's DTM in the frame of the
grid, which is UTM minus the cloud's mean M (see the README's Data section),
so the mesh is built in that frame. The transform used is the one inside the
TIFF, which is what GDAL, QGIS and ParaView use; a .tfw beside it is ignored,
and at Rialba the two disagree by 38 m in x.
"""

import sys
import time
from pathlib import Path

import meshio
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from . import terrain as terrain_surfaces
from .case import load_case
from .dtm_io import interpolate_in_parallel
from .shift_fracture import M  # the same Rialba mean
from .terrain import compact_triangles, grid_triangles

TARGET_PIXELS = 2_000_000


def downsample(path):
    """(rgba (h, w, 4), transform) of the photo, decimated to TARGET_PIXELS."""
    with rasterio.open(path) as photo:
        factor = max(1, int(np.ceil(np.sqrt(photo.width * photo.height / TARGET_PIXELS))))
        height, width = photo.height // factor, photo.width // factor
        print(
            f"read {path}: {photo.width:,} x {photo.height:,} pixels, {photo.count} bands, "
            f"{photo.crs}, {photo.res[0]:.3f} m/pixel\n"
            f"  averaging {factor} x {factor} pixels -> {width:,} x {height:,}, "
            f"{factor * photo.res[0]:.3f} m/pixel",
            flush=True,
        )
        t0 = time.perf_counter()
        bands = photo.read(out_shape=(photo.count, height, width), resampling=Resampling.average)
        print(f"  decimated read in {time.perf_counter() - t0:.1f}s", flush=True)
        return np.moveaxis(bands, 0, -1), photo.transform * rasterio.Affine.scale(factor)


def drape(rgba, transform, terrain):
    """(points, triangles, rgb) of the photo's pixels, lifted onto the terrain."""
    height, width, _ = rgba.shape
    # pixel centres, in the frame of the grid; rows go north to south, so the
    # image is flipped to let both the mesh's j and y increase together
    rgba = rgba[::-1]
    east = transform.c + (np.arange(width) + 0.5) * transform.a - M[0]
    north = transform.f + (np.arange(height - 1, -1, -1) + 0.5) * transform.e - M[1]
    x, y = np.meshgrid(east, north, indexing="ij")

    t0 = time.perf_counter()
    z = interpolate_in_parallel(terrain.interp, np.column_stack([x.ravel(), y.ravel()]))
    print(
        f"  terrain interpolated at {x.size:,} pixel centres in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    points = np.column_stack([x.ravel(), y.ravel(), z])
    rgb = np.moveaxis(rgba[:, :, :3], 0, 1).reshape(-1, 3)
    opaque = np.moveaxis(rgba[:, :, 3], 0, 1).ravel() > 0
    triangles = grid_triangles(width, height)
    keep = opaque[triangles].all(axis=1) & np.isfinite(z[triangles]).all(axis=1)
    points, triangles, used = compact_triangles(points, triangles, keep)
    print(
        f"  {len(triangles):,} of {len(keep):,} triangles have all their corners on an "
        f"opaque pixel over the DTM, on {len(points):,} of {len(keep) // 2 + width + height:,} "
        f"pixels",
        flush=True,
    )
    return points, triangles, rgb[used]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    photo_path = Path(argv[0]).resolve()
    case = load_case(argv[1])
    if case.surface != "height_field":
        raise SystemExit(f"{argv[1]}: surface is {case.surface!r}, draping needs a height field")

    rgba, transform = downsample(photo_path)
    small = photo_path.with_name(photo_path.stem + "_small.png")
    Image.fromarray(rgba).save(small)
    print(f"wrote {small}: {small.stat().st_size / 1024**2:.1f} MB", flush=True)

    terrain, _, _ = terrain_surfaces.build(case)
    points, triangles, rgb = drape(rgba, transform, terrain)
    draped = photo_path.with_suffix(".vtu")
    meshio.write_points_cells(
        draped, points, [("triangle", triangles)], point_data={"rgb": rgb}, binary=True
    )
    print(
        f"wrote {draped}: {len(points):,} points, {len(triangles):,} triangles, "
        f"{draped.stat().st_size / 1024**2:.1f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
