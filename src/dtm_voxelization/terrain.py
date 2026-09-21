"""The terrain surface a grid is carved against, in the grid's frame.

Two kinds, chosen by the case's `surface`:

- HeightField: the terrain is z = f(x, y), the linear interpolant of the
  points over their Delaunay triangulation in (x, y). A hex is kept when its
  top lies below the terrain at its (x, y) centroid.
- PoissonSurface: any surface, overhangs and re-entrances included,
  reconstructed from the points by screened Poisson (Open3D, imported only
  here). A hex is kept when the top of its centroid's vertical segment is in
  the rock -- an odd number of surface crossings straight above it -- and the
  segment down to the hex bottom crosses nothing, i.e. no overhang cuts the
  hex. On a surface that is a height field this is the same rule as above.

Both answer what the grid asks: the domain rectangle, the surface's z range,
the rock fraction of any box (for sizing), the terrain's z range over a
refinement rectangle, which hexes to keep, and an STL of the surface.
"""

import time

import meshio
import numpy as np
from scipy import ndimage

from . import frame
from .dtm_io import (
    _load_xyz,
    interpolate_in_parallel,
    load_dtm_analysis_grid,
    load_dtm_interpolator,
)

FRACTION_SAMPLE_N = 400  # resolution of the coarse pre-sample used only to size the grid
FOOTPRINT_BINS = 300  # bins along the longer side when locating the footprint's rectangle
RAY_OFFSET = (1.3e-4, 2.9e-4)  # skew (x, y) nudge of vertical rays off triangle edges
RAY_CHUNK = 2_000_000


def largest_rectangle(mask):
    """Largest all-True axis-aligned block of a 2D mask: (i0, i1, j0, j1), half-open."""
    nx, ny = mask.shape
    heights = np.zeros(ny, dtype=int)
    best_area, best = 0, None
    for i in range(nx):
        heights = np.where(mask[i], heights + 1, 0)
        stack = []
        for j in range(ny + 1):
            h = heights[j] if j < ny else 0
            start = j
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                if sh * (j - s) > best_area:
                    best_area, best = sh * (j - s), (i - sh + 1, i + 1, s, j)
                start = s
            stack.append((start, h))
    return best


def footprint_rectangle(xy):
    """Largest rectangle inside the points' (x, y) footprint, interior holes
    filled, one bin in from every side."""
    lo = xy.min(axis=0)
    size = np.ptp(xy, axis=0).max() / FOOTPRINT_BINS
    ij = np.floor((xy - lo) / size).astype(int)
    mask = np.zeros(ij.max(axis=0) + 1, dtype=bool)
    mask[ij[:, 0], ij[:, 1]] = True
    filled = ndimage.binary_fill_holes(mask)
    i0, i1, j0, j1 = largest_rectangle(filled)
    rect = (lo[0] + (i0 + 1) * size, lo[0] + (i1 - 1) * size, lo[1] + (j0 + 1) * size, lo[1] + (j1 - 1) * size)
    print(
        f"  footprint: {int(mask.sum()):,} bins of {size:.3f} covered, {int((filled & ~mask).sum()):,} "
        f"interior holes filled, {100 * filled.mean():.1f}% of the bbox; rectangle "
        f"x=[{rect[0]:.2f},{rect[1]:.2f}] y=[{rect[2]:.2f},{rect[3]:.2f}], "
        f"{100 * (i1 - i0 - 2) * (j1 - j0 - 2) / filled.size:.1f}% of the bbox",
        flush=True,
    )
    return rect


def compact_triangles(points, triangles, keep):
    """Only the kept triangles, renumbered onto the points they use, and the
    indices of those points, for whatever else is attached to them."""
    used, compact = np.unique(triangles[keep], return_inverse=True)
    return points[used], compact.reshape(-1, 3), used


def grid_triangles(nx, ny):
    """Two triangles per quad of an nx x ny grid of vertices numbered i * ny + j,
    counter-clockwise seen from +z."""
    i = np.arange(nx - 1)[:, None]
    j = np.arange(ny - 1)[None, :]
    lower_left = (i * ny + j).ravel()
    lower_right = lower_left + ny
    upper_left = lower_left + 1
    upper_right = lower_right + 1
    return np.vstack(
        (
            np.column_stack((lower_left, lower_right, upper_right)),
            np.column_stack((lower_left, upper_right, upper_left)),
        )
    )


def export_triangles(path, points, triangles, keep=None):
    """Binary STL: an ASCII one of a Poisson mesh runs to hundreds of MB.
    `keep`, a mask over the triangles, writes only those and their points.
    Zero-area triangles, which a Poisson mesh has a few of, are dropped: their
    normal would be written as NaN."""
    corners = points[triangles]
    area2 = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    degenerate = area2 == 0
    if degenerate.any():
        print(f"  dropping {int(degenerate.sum()):,} zero-area triangles", flush=True)
        keep = ~degenerate if keep is None else keep & ~degenerate
    if keep is not None:
        points, triangles, _ = compact_triangles(points, triangles, keep)
    meshio.write_points_cells(path, points, [("triangle", triangles)], binary=True)
    print(f"wrote {path} -- {len(points):,} points, {len(triangles):,} triangles", flush=True)


class HeightField:
    """z = f(x, y), linear over the Delaunay triangulation of the points."""

    def __init__(self, points_path, trim_to_footprint):
        self.dtm = load_dtm_analysis_grid(points_path)
        self.interp = load_dtm_interpolator(points_path)
        self.xmin, self.xmax = self.dtm["xmin"], self.dtm["xmax"]
        self.ymin, self.ymax = self.dtm["ymin"], self.dtm["ymax"]
        if trim_to_footprint:
            x, y, _ = _load_xyz(points_path)
            self.xmin, self.xmax, self.ymin, self.ymax = footprint_rectangle(np.column_stack([x, y]))
        self.zmin = float(np.nanmin(self.dtm["Z"]))
        self.zmax = float(np.nanmax(self.dtm["Z"]))

    def kept_fraction(self, xmin, xmax, ymin, ymax, zmin, lz, n=FRACTION_SAMPLE_N):
        """Fraction of the box below the terrain, from a coarse sample."""
        xs = xmin + (np.arange(n) + 0.5) * (xmax - xmin) / n
        ys = ymin + (np.arange(n) + 0.5) * (ymax - ymin) / n
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        terrain = self.interp(np.column_stack([xx.ravel(), yy.ravel()])).reshape(n, n)
        frac = np.clip((terrain - zmin) / lz, 0, 1)
        return float(np.mean(frac))

    def z_extent(self, rxmin, rxmax, rymin, rymax, pad, n=50):
        """[terrain min - pad, terrain max + pad] within the rectangle."""
        xs = np.linspace(rxmin, rxmax, n)
        ys = np.linspace(rymin, rymax, n)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        terrain = self.interp(np.column_stack([xx.ravel(), yy.ravel()]))
        return float(np.nanmin(terrain)) - pad, float(np.nanmax(terrain)) + pad

    def keep(self, coords):
        t0 = time.perf_counter()
        centroids = coords[:, :, :2].mean(axis=1)
        terrain_at_centroid = interpolate_in_parallel(self.interp, centroids)
        print(
            f"[carve] terrain interpolated at {len(centroids):,} centroids in "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return coords[:, :, 2].max(axis=1) <= terrain_at_centroid

    def export(self, path):
        """The regular DTM analysis grid as an upward-facing STL surface."""
        x_grid, y_grid, z_grid = self.dtm["X"], self.dtm["Y"], self.dtm["Z"]
        nx, ny = z_grid.shape
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
        export_triangles(path, points, grid_triangles(nx, ny))


class PoissonSurface:
    """Any surface, reconstructed from the points by screened Poisson."""

    def __init__(self, points, voxel_size, depth, trim_to_footprint):
        import open3d as o3d

        self.o3d = o3d
        t0 = time.perf_counter()
        self.points = points
        if trim_to_footprint:
            self.xmin, self.xmax, self.ymin, self.ymax = footprint_rectangle(points[:, :2])
        else:
            (self.xmin, self.ymin), (self.xmax, self.ymax) = points[:, :2].min(0), points[:, :2].max(0)
        inside = self._in_rectangle(self.xmin, self.xmax, self.ymin, self.ymax)
        self.zmin, self.zmax = float(points[inside, 2].min()), float(points[inside, 2].max())

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(voxel_size)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel_size, max_nn=30))
        cloud.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
        # propagate the orientation along the surface: under an overhang the
        # true normal points down, which a fixed direction would get wrong
        cloud.orient_normals_consistent_tangent_plane(15)
        # np.asarray views Open3D's buffer, which reassigning the normals frees
        if np.asarray(cloud.normals)[:, 2].mean() < 0:
            cloud.normals = o3d.utility.Vector3dVector(-np.asarray(cloud.normals).copy())
        normals = np.asarray(cloud.normals).copy()
        print(
            f"  {len(points):,} points -> {len(cloud.points):,} after a {voxel_size} voxel down-sample; "
            f"normals pointing down (overhang undersides): {100 * (normals[:, 2] < 0).mean():.2f}%",
            flush=True,
        )

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=depth, scale=1.1, linear_fit=False
        )
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        print(
            f"  poisson depth {depth}: {len(self.vertices):,} vertices, {len(self.triangles):,} triangles, "
            f"built in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    def _in_rectangle(self, rxmin, rxmax, rymin, rymax):
        x, y = self.points[:, 0], self.points[:, 1]
        return (x >= rxmin) & (x <= rxmax) & (y >= rymin) & (y <= rymax)

    def crossings_above(self, xy, z):
        """Surface crossings along the vertical ray up from each (x, y, z)."""
        counts = np.empty(len(z), dtype=np.int64)
        for lo in range(0, len(z), RAY_CHUNK):
            hi = min(lo + RAY_CHUNK, len(z))
            rays = np.zeros((hi - lo, 6), dtype=np.float32)
            rays[:, 0] = xy[lo:hi, 0] + RAY_OFFSET[0]
            rays[:, 1] = xy[lo:hi, 1] + RAY_OFFSET[1]
            rays[:, 2] = z[lo:hi]
            rays[:, 5] = 1.0
            counts[lo:hi] = self.scene.count_intersections(self.o3d.core.Tensor(rays)).numpy()
        return counts

    def kept_fraction(self, xmin, xmax, ymin, ymax, zmin, lz, n=60):
        """Fraction of the box in the rock, from parity at a coarse sample."""
        xs = xmin + (np.arange(n) + 0.5) * (xmax - xmin) / n
        ys = ymin + (np.arange(n) + 0.5) * (ymax - ymin) / n
        zs = zmin + (np.arange(n) + 0.5) * lz / n
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        counts = self.crossings_above(np.column_stack([xx.ravel(), yy.ravel()]), zz.ravel())
        return float(np.mean(counts % 2 == 1))

    def z_extent(self, rxmin, rxmax, rymin, rymax, pad):
        """[lowest point - pad, highest point + pad] within the rectangle."""
        inside = self._in_rectangle(rxmin, rxmax, rymin, rymax)
        if not inside.any():
            raise ValueError("the refinement rectangle contains no DTM points")
        z = self.points[inside, 2]
        return float(z.min()) - pad, float(z.max()) + pad

    def keep(self, coords):
        t0 = time.perf_counter()
        xy = coords[:, :, :2].mean(axis=1)
        above_top = self.crossings_above(xy, coords[:, :, 2].max(axis=1))
        above_bottom = self.crossings_above(xy, coords[:, :, 2].min(axis=1))
        in_rock = above_top % 2 == 1
        cut_by_surface = above_bottom != above_top
        print(
            f"[carve] {2 * len(xy):,} vertical rays in {time.perf_counter() - t0:.1f}s: top in rock "
            f"{int(in_rock.sum()):,}, of which crossed by the surface {int((in_rock & cut_by_surface).sum()):,}",
            flush=True,
        )
        return in_rock & ~cut_by_surface

    def export(self, path):
        """The part of the mesh over the domain rectangle: outside the data the
        reconstruction closes itself with surfaces of no meaning, which the
        carving never meets but which would bury the terrain in ParaView."""
        centroids = self.vertices[self.triangles].mean(axis=1)
        inside = (
            (centroids[:, 0] >= self.xmin)
            & (centroids[:, 0] <= self.xmax)
            & (centroids[:, 1] >= self.ymin)
            & (centroids[:, 1] <= self.ymax)
        )
        print(
            f"  surface over the domain rectangle: {int(inside.sum()):,} of "
            f"{len(self.triangles):,} triangles",
            flush=True,
        )
        export_triangles(path, self.vertices, self.triangles, keep=inside)


def build(case):
    """(terrain, rotation, center) for the case, the terrain in the grid's frame."""
    if case.surface == "height_field":
        return HeightField(case.points, case.trim_to_footprint), np.eye(3), np.zeros(3)

    x, y, z = _load_xyz(case.points)
    return poisson_surface(np.column_stack([x, y, z]), case)


def poisson_surface(points, case):
    """(surface, rotation, center): the case's Poisson reconstruction of the
    points, in the frame where their least-squares plane is horizontal."""
    rotation, center = frame.compute(points, case.outward)
    print(
        f"  frame: {'identity' if frame.is_identity(rotation, center) else 'least-squares plane'}, "
        f"normal {rotation[2].round(4)}, center {center.round(3)}",
        flush=True,
    )
    surface = PoissonSurface(
        frame.to_grid(points, rotation, center),
        case.voxel_size,
        case.poisson_depth,
        case.trim_to_footprint,
    )
    return surface, rotation, center
