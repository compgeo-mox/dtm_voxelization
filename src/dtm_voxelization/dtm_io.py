"""Minimal DTM I/O for build_cartgrid_carved.py: a coarse resampled
analysis grid (domain extent + elevation, for sizing the Cartesian grid)
and a full-resolution scattered-point interpolator (for querying exact
terrain elevation while carving)."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata


def _load_xyz(file_path):
    data = np.loadtxt(file_path, skiprows=2)
    x_col, y_col, z_col = (1, 2, 3) if data.shape[1] >= 4 else (0, 1, 2)
    return data[:, x_col], data[:, y_col], data[:, z_col]


def load_dtm_analysis_grid(xyz_path, max_grid_points=300):
    """Resampled analysis grid: a coarse regular (x, y) grid with
    interpolated elevation, plus the DTM's own (x, y) extent. Coarse by
    design -- for domain sizing only, not full-accuracy carving (see
    load_dtm_interpolator for that)."""
    t0 = time.time()
    x, y, z = _load_xyz(xyz_path)
    print(f"  [loadtxt] {time.time() - t0:.2f}s  shape=({len(x)}, 3)")

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_step = max(1, len(x_unique) // max_grid_points)
    y_step = max(1, len(y_unique) // max_grid_points)
    x_unique = x_unique[::x_step]
    y_unique = y_unique[::y_step]
    print(f"  [grid] resampled to {len(x_unique)}x{len(y_unique)} points")

    t0 = time.time()
    X, Y = np.meshgrid(x_unique, y_unique, indexing="ij")
    Z = griddata((x, y), z, (X, Y), method="linear")
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        ix, iy = np.where(~nan_mask)
        nn = NearestNDInterpolator(list(zip(ix, iy)), Z[~nan_mask])
        ni, nj = np.where(nan_mask)
        Z[nan_mask] = nn(list(zip(ni, nj)))
    print(f"  [griddata] {time.time() - t0:.2f}s")

    return dict(
        X=X,
        Y=Y,
        Z=Z,
        x_unique=x_unique,
        y_unique=y_unique,
        xmin=float(x_unique.min()),
        xmax=float(x_unique.max()),
        ymin=float(y_unique.min()),
        ymax=float(y_unique.max()),
    )


def load_dtm_interpolator(xyz_path):
    """Native-resolution DTM lookup: a LinearNDInterpolator triangulated
    once over the full raw point cloud, with NearestNDInterpolator as a
    fallback outside the point cloud's convex hull (where the linear
    interpolator returns NaN). Returns a callable: query_xy (N, 2) array
    -> z (N,) array."""
    t0 = time.time()
    x, y, z = _load_xyz(xyz_path)
    print(f"  [loadtxt] {time.time() - t0:.2f}s  shape=({len(x)}, 3)")

    t0 = time.time()
    xy = np.column_stack([x, y])
    linear = LinearNDInterpolator(xy, z)
    nearest = NearestNDInterpolator(xy, z)
    print(f"  [triangulate] {time.time() - t0:.2f}s")

    def interpolate(query_xy):
        z_out = linear(query_xy)
        nan_mask = np.isnan(z_out)
        if nan_mask.any():
            z_out[nan_mask] = nearest(query_xy[nan_mask])
        return z_out

    return interpolate


def interpolate_in_parallel(
    interpolator, query_xy, min_points=100_000, max_workers=None
):
    """Evaluate a DTM interpolator in independent chunks when worthwhile.

    SciPy's interpolators release the GIL during their numerical work, while
    the callable returned by ``load_dtm_interpolator`` is safe to share for
    read-only queries. Small arrays stay on the simple single-call path so
    thread setup never dominates the work.
    """
    query_xy = np.asarray(query_xy)
    if len(query_xy) < min_points:
        return interpolator(query_xy)

    workers = max_workers or min(4, os.cpu_count() or 1)
    workers = max(1, min(workers, len(query_xy)))
    chunks = np.array_split(query_xy, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(interpolator, chunks))
    return np.concatenate(results)
