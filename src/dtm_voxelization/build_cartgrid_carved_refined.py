"""Like build_cartgrid_carved.py, but with two NESTED regions locally
refined -- 3x3x3 hex split plus the wall/corner/concave transition layers
from general_rebuild.py's template scheme -- before the terrain carving.

The inner region (INNER_X/Y_MIN/MAX) gets a DOUBLE refinement (level 1,
then a further nested level-2 split inside it); the outer region is that
same rectangle scaled by OUTER_SCALE in x, y AND z, same center, and gets
only the level-1 split -- it exists to give the inner region's level-2
split the buffer ring of level-1 'urz' cells it needs (see
general_rebuild.refine_region_further's docstring).

Unlike the plain CartGrid pipeline, the refinement templates require an
ISOTROPIC x/y cell size (only z may differ, see templates.apply_matrix's
docstring), so NX/NY/h_xy are locked together here instead of chosen
independently -- the domain's x/y extent is padded out to an exact
multiple of h_xy to make that possible.

TARGET_TOTAL_CELLS sizing: build_cartgrid_carved.py's own cell_edge
formula assumes a PLAIN grid (1 hex/cell); refinement replaces cells with
27/13/5-hex templates, so that formula alone underestimates the final
count once the two nested regions are added. Rather than modeling that
inflation analytically (it depends on how many coarse cells the fixed
physical regions cover, which itself depends on the cell size being
solved for), `main` just measures it directly: build+carve once with the
plain-grid estimate, then rescale cell_edge by the observed ratio to
target and rebuild -- reusing the exact same build/carve code each time.

The refined mesh is a plain (points, hexes, tags) hex soup (general_rebuild's
own output), not a PorePy grid, so carving and export are done directly
here instead of via pp.CartGrid/pp.partition.extract_subgrid.
"""

from pathlib import Path

import numpy as np
import meshio

from .dtm_io import load_dtm_analysis_grid, load_dtm_interpolator
from .build_cartgrid_carved import TARGET_TOTAL_CELLS, estimate_kept_fraction
from . import general_rebuild as GR
from .geometry_checks import check_conformity_3d

REPO_ROOT = Path(__file__).resolve().parents[2]
XYZ_PATH = REPO_ROOT / "data" / "xyz" / "merged.xyz"
OUTPUT_DIR = REPO_ROOT / "output"

# inner (finest) region, hand-picked physical (x, y) rectangle (from the
# DTM slope map's layer-3 nested refinement, test_gridding/files/manual_regions.py)
INNER_XMIN, INNER_XMAX = -260.0, -10.0
INNER_YMIN, INNER_YMAX = 175.0, 425.0
REGION_Z_PAD = 20.0  # extra clearance above/below the DTM surface within the inner region

OUTER_SCALE = 1.5  # outer (level-1) region = inner region scaled by this in x, y, AND z, same center
SIZE_TOLERANCE = 0.3  # accept the first cell_edge landing within +/-30% of target_total_cells


def footprint_from_region(xmin, ymin, hx, hy, nx, ny, rxmin, rxmax, rymin, rymax):
    """Coarse-grid (i,j) cells whose CENTER falls inside the rectangle --
    same convention as test_gridding's manual_regions.rect_mask."""
    cx = xmin + (np.arange(nx) + 0.5) * hx
    cy = ymin + (np.arange(ny) + 0.5) * hy
    ii = np.where((cx >= rxmin) & (cx <= rxmax))[0]
    jj = np.where((cy >= rymin) & (cy <= rymax))[0]
    return {(int(i), int(j)) for i in ii for j in jj}


def z_extent_from_region(dtm_interp, rxmin, rxmax, rymin, rymax, pad, n=50):
    """[DTM min - pad, DTM max + pad] within the rectangle, sampled from
    the full-accuracy interpolator."""
    xs = np.linspace(rxmin, rxmax, n)
    ys = np.linspace(rymin, rymax, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    terrain = dtm_interp(np.column_stack([xx.ravel(), yy.ravel()]))
    return float(np.nanmin(terrain)) - pad, float(np.nanmax(terrain)) + pad


def k_range_from_z(zbot, ztop, zmin, hz, nz):
    """Consecutive z-layer index range (at spacing hz) covering [zbot, ztop]."""
    k_lo = max(0, int(np.floor((zbot - zmin) / hz)))
    k_hi = min(nz - 1, int(np.ceil((ztop - zmin) / hz)) - 1)
    k_hi = max(k_hi, k_lo)
    return list(range(k_lo, k_hi + 1))


def scale_interval(lo, hi, scale):
    """Grows/shrinks [lo, hi] by `scale` about its own center."""
    c, h = (lo + hi) / 2, (hi - lo) / 2 * scale
    return c - h, c + h


def build_double_refined_grid(cell_edge, XMIN, YMIN, ZMIN, Lx_dtm, Ly_dtm, Lz,
                               inner_bounds, outer_bounds, dtm_interp):
    """One full build attempt at the given (isotropic-xy) cell_edge: grid
    sizing, both nested regions' footprints/k-layers, level-1 build_mesh,
    level-2 refine_region_further, and carving. Returns (points, hexes,
    tags, keep, grid_info) -- grid_info is a dict of everything printed/
    reused by the caller (NX, NY, NZ, h_xy, hz, XMAX, YMAX, footprint
    sizes, conformity results).

    general_rebuild's point-dedup rounds position/cell_size to 5 decimal
    places (see build_mesh's gid() docstring) -- fine for the small,
    origin-relative offsets its own tests use, but this DTM's real
    XMIN/YMIN/ZMIN are large, mutually different, non-round numbers, and
    an unlucky combination can land a legitimate shared point right on a
    rounding-tie boundary, splitting it into two near-duplicates (a real,
    reproduced hanging-node/phantom-face defect). Building in a LOCAL
    zero-based frame (only the grid's own indices matter for topology)
    and translating the finished points by the true offset afterward
    sidesteps this entirely, since a constant translation can't change
    which points coincide.
    """
    NX = max(1, round(Lx_dtm / cell_edge))
    NY = max(1, round(Ly_dtm / cell_edge))
    NZ = max(1, round(Lz / cell_edge))
    h_xy = cell_edge
    hz = Lz / NZ
    XMAX = XMIN + NX * h_xy  # padded slightly past the DTM's real extent
    YMAX = YMIN + NY * h_xy  # so hx == hy exactly (required by the templates)
    ZMAX = ZMIN + NZ * hz
    LX0 = LY0 = LZ0 = 0.0  # local build frame; translated to (XMIN,YMIN,ZMIN) at the end

    (inner_xmin, inner_xmax, inner_ymin, inner_ymax, inner_zbot, inner_ztop) = inner_bounds
    (outer_xmin, outer_xmax, outer_ymin, outer_ymax, outer_zbot, outer_ztop) = outer_bounds

    footprint1 = footprint_from_region(
        XMIN, YMIN, h_xy, h_xy, NX, NY, outer_xmin, outer_xmax, outer_ymin, outer_ymax)
    if not footprint1:
        raise ValueError("outer region does not overlap any coarse cell center")
    k_layers1 = k_range_from_z(outer_zbot, outer_ztop, ZMIN, hz, NZ)

    points, hexes, tags = GR.build_mesh(
        footprint1, NX=NX, NY=NY, NZ=NZ, XMIN=LX0, YMIN=LY0, ZMIN=LZ0,
        k_layers=k_layers1, cell_size=(h_xy, h_xy, hz))
    ok1, detail1 = GR.verify_mesh(points, hexes, footprint1, NX, NY, LX0, NX * h_xy, LY0, NY * h_xy,
                                   LZ0, NZ * hz, n_layers=len(k_layers1))
    if not ok1:
        raise RuntimeError(f"level-1 refined mesh is not conforming: {detail1}")

    # level 2: the inner region, one scale finer, nested inside the
    # level-1 urz block (strictly inside the outer region by construction).
    NXf, NYf, NZf = NX * 3, NY * 3, NZ * 3
    hxf, hyf, hzf = h_xy / 3, h_xy / 3, hz / 3
    footprint2 = footprint_from_region(
        XMIN, YMIN, hxf, hyf, NXf, NYf, inner_xmin, inner_xmax, inner_ymin, inner_ymax)
    if not footprint2:
        raise ValueError("inner region does not overlap any level-1-child cell center")
    buf_lo = k_layers1[0] * 3 + 2   # +2: mandatory same-scale buffer (local_size=5, buf=2)
    buf_hi = (k_layers1[-1] + 1) * 3 - 1 - 2
    k_layers2 = [k for k in k_range_from_z(inner_zbot, inner_ztop, ZMIN, hzf, NZf)
                 if buf_lo <= k <= buf_hi]
    if not k_layers2:
        raise ValueError("inner region's z range leaves no room for the level-2 vertical buffer")

    points, hexes, tags = GR.refine_region_further(
        points, hexes, tags, footprint2, k_layers2, NXf, NYf, NZf, LX0, LY0, LZ0,
        cell_size=(hxf, hyf, hzf))
    ok2 = check_conformity_3d(points, hexes, LX0, NX * h_xy, LY0, NY * h_xy, LZ0, NZ * hz)
    if not ok2:
        raise RuntimeError("double-refined mesh is not conforming")

    points = points + np.array([XMIN, YMIN, ZMIN])

    # carve: drop every hex entirely above the terrain at its own (x,y)
    # centroid -- same rule as build_cartgrid_carved.py's `keep`.
    coords = points[hexes]  # (n_hex, 8, 3)
    terrain_at_centroid = dtm_interp(coords[:, :, :2].mean(axis=1))
    keep = coords[:, :, 2].max(axis=1) <= terrain_at_centroid

    grid_info = dict(
        NX=NX, NY=NY, NZ=NZ, h_xy=h_xy, hz=hz, XMAX=XMAX, YMAX=YMAX,
        n_footprint1=len(footprint1), k_layers1=k_layers1,
        n_footprint2=len(footprint2), k_layers2=k_layers2,
        detail1=detail1)
    return points, hexes, tags, keep, grid_info


def main(target_total_cells=TARGET_TOTAL_CELLS, outer_scale=OUTER_SCALE, max_iterations=3):
    dtm = load_dtm_analysis_grid(XYZ_PATH)
    XMIN, YMIN = dtm["xmin"], dtm["ymin"]
    ZMIN = float(np.nanmin(dtm["Z"])) - 50.0
    ZMAX = float(np.nanmax(dtm["Z"])) + 50.0
    Lz = ZMAX - ZMIN
    Lx_dtm = dtm["xmax"] - XMIN
    Ly_dtm = dtm["ymax"] - YMIN

    dtm_interp = load_dtm_interpolator(XYZ_PATH)
    kept_fraction = estimate_kept_fraction(
        dtm_interp, XMIN, dtm["xmax"], YMIN, dtm["ymax"], ZMIN, Lz)
    print(f"estimated below-terrain (kept) fraction of the box: {kept_fraction:.4f}")

    # the two nested regions' physical bounds don't depend on grid
    # resolution -- computed once, outside the cell_edge calibration loop.
    inner_zbot, inner_ztop = z_extent_from_region(
        dtm_interp, INNER_XMIN, INNER_XMAX, INNER_YMIN, INNER_YMAX, REGION_Z_PAD)
    inner_bounds = (INNER_XMIN, INNER_XMAX, INNER_YMIN, INNER_YMAX, inner_zbot, inner_ztop)
    outer_bounds = (
        *scale_interval(INNER_XMIN, INNER_XMAX, outer_scale),
        *scale_interval(INNER_YMIN, INNER_YMAX, outer_scale),
        *scale_interval(inner_zbot, inner_ztop, outer_scale))
    print(f"inner region: x=[{INNER_XMIN:.1f},{INNER_XMAX:.1f}] "
          f"y=[{INNER_YMIN:.1f},{INNER_YMAX:.1f}] z=[{inner_zbot:.1f},{inner_ztop:.1f}]")
    print(f"outer region ({outer_scale}x, same center): "
          f"x=[{outer_bounds[0]:.1f},{outer_bounds[1]:.1f}] "
          f"y=[{outer_bounds[2]:.1f},{outer_bounds[3]:.1f}] "
          f"z=[{outer_bounds[4]:.1f},{outer_bounds[5]:.1f}]")

    # plain-grid estimate (ignores refinement inflation), then calibrated
    # against the actual built+carved cell count below.
    cell_edge = (Lx_dtm * Ly_dtm * Lz * kept_fraction / target_total_cells) ** (1.0 / 3.0)
    for iteration in range(max_iterations):
        points, hexes, tags, keep, grid = build_double_refined_grid(
            cell_edge, XMIN, YMIN, ZMIN, Lx_dtm, Ly_dtm, Lz,
            inner_bounds, outer_bounds, dtm_interp)
        n_kept = int(keep.sum())
        ratio = n_kept / target_total_cells
        print(f"[iter {iteration}] cell_edge={cell_edge:.3f} -> grid "
              f"{grid['NX']}x{grid['NY']}x{grid['NZ']}, level 1 "
              f"{grid['n_footprint1']} cells / level 2 {grid['n_footprint2']} cells, "
              f"{len(hexes):,} hexes, {n_kept:,} kept (target {target_total_cells:,}, "
              f"ratio {ratio:.2f})")
        if abs(ratio - 1.0) <= SIZE_TOLERANCE or iteration == max_iterations - 1:
            break
        cell_edge *= ratio ** (1.0 / 3.0)  # refinement inflates cell count beyond the plain-grid estimate

    print(f"level 1 conformity check: {grid['detail1']}")
    print("level 2 (combined) conformity check: PASS")

    kept_hexes = hexes[keep]
    kind_str = [t.split(":")[0] for t, k in zip(tags, keep) if k]
    kind_names = sorted(set(kind_str))
    kind_code = np.array([kind_names.index(k) for k in kind_str], dtype=float)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "cartgrid_carved_refined.vtu"
    meshio.write_points_cells(
        out_path, points, [("hexahedron", kept_hexes)],
        cell_data={"kind_code": [kind_code]})
    print(f"wrote {out_path} -- kind_code legend: {list(enumerate(kind_names))}")


if __name__ == "__main__":
    main()
