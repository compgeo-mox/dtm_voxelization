"""Refined, terrain-carved hex grid of a DTM.

A Cartesian grid over the DTM's extent with two NESTED regions locally
refined -- 3x3x3 hex split plus the wall/corner/concave transition layers
from general_rebuild.py's template scheme -- then carved: every hex not in
the rock is dropped. What "in the rock" means is the terrain's business
(terrain.py), and so is the frame the grid is built in (frame.py).

The inner region (the case's `inner_region`) gets a DOUBLE refinement (level
1, then a further nested level-2 split inside it); the outer region is that
same rectangle scaled by `outer_scale` in x, y AND z, same center, and gets
only the level-1 split -- it exists to give the inner region's level-2 split
the buffer ring of level-1 'urz' cells it needs (see
general_rebuild.refine_region_further's docstring). So in x/y it is never
less than 1.5 coarse cells wider than the inner region on every side, whatever
the scale: see `outer_region`.

The refinement templates require an ISOTROPIC x/y cell size (only z may
differ, see templates.apply_matrix's docstring), so NX/NY/h_xy are locked
together -- the domain's x/y extent is padded out to an exact multiple of
h_xy to make that possible.

Sizing to `target_cells` happens twice. First without building anything:
`predicted_cells` counts the coarse cells, plus 26 more hexes per refined cell
of each level, each weighted by the rock fraction of its region, and
cell_edge is rescaled until that count meets the target. This matters because
a refined region that is large for the domain inflates the count by one to
two orders of magnitude, and a first build sized on the plain grid alone
would be that much too big -- enough to exhaust memory inside the level-2
split before any correction could happen. Then the real builds: build+carve,
rescale cell_edge by the observed ratio to target (the prediction ignores the
transition templates) and rebuild, reusing the exact same code each time.
"""

import time

import numpy as np
import meshio

from . import frame
from . import general_rebuild as GR
from . import terrain as terrain_surfaces
from .geometry_checks import check_conformity_3d
from .mesh import peak_memory_gb

SIZE_TOLERANCE = 0.3  # accept the first cell_edge landing within +/-30% of target_cells
MAX_ITERATIONS = 3
PRESIZE_ITERATIONS = 30


def footprint_from_region(xmin, ymin, hx, hy, nx, ny, rxmin, rxmax, rymin, rymax):
    """Coarse-grid (i,j) cells whose CENTER falls inside the rectangle --
    same convention as test_gridding's manual_regions.rect_mask."""
    cx = xmin + (np.arange(nx) + 0.5) * hx
    cy = ymin + (np.arange(ny) + 0.5) * hy
    ii = np.where((cx >= rxmin) & (cx <= rxmax))[0]
    jj = np.where((cy >= rymin) & (cy <= rymax))[0]
    return {(int(i), int(j)) for i in ii for j in jj}


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


def outer_region(inner_bounds, outer_scale, cell_edge):
    """Level-1 region for a build at `cell_edge`: the inner region scaled by
    `outer_scale` about its center, and in x/y at least 1.5 coarse cells wider
    on every side. The level-2 buffer reaches 2/3 of a coarse cell past the
    inner region, a coarse cell is refined only when its center is inside, and
    a child's center is within 1/3 of a cell of its parent's: 1 cell of margin
    is the least that always works, 1.5 keeps clear of rounding ties."""
    xmin, xmax, ymin, ymax, zbot, ztop = inner_bounds
    sxmin, sxmax = scale_interval(xmin, xmax, outer_scale)
    symin, symax = scale_interval(ymin, ymax, outer_scale)
    margin = 1.5 * cell_edge
    return (
        min(sxmin, xmin - margin),
        max(sxmax, xmax + margin),
        min(symin, ymin - margin),
        max(symax, ymax + margin),
        *scale_interval(zbot, ztop, outer_scale),
    )


def predicted_cells(cell_edge, XMIN, YMIN, ZMIN, Lx_dtm, Ly_dtm, Lz, inner_bounds, outer_scale, fractions):
    """Carved cells a build at `cell_edge` would give, without building it:
    the same grid sizing, footprints and layers as build_double_refined_grid,
    with each level's refined cells adding 26 hexes weighted by the rock
    fraction of its region. Transition templates are ignored."""
    NX = max(1, round(Lx_dtm / cell_edge))
    NY = max(1, round(Ly_dtm / cell_edge))
    NZ = max(1, round(Lz / cell_edge))
    hz = Lz / NZ
    outer_bounds = outer_region(inner_bounds, outer_scale, cell_edge)
    footprint1 = footprint_from_region(XMIN, YMIN, cell_edge, cell_edge, NX, NY, *outer_bounds[:4])
    k_layers1 = k_range_from_z(outer_bounds[4], outer_bounds[5], ZMIN, hz, NZ)
    footprint2 = footprint_from_region(
        XMIN, YMIN, cell_edge / 3, cell_edge / 3, 3 * NX, 3 * NY, *inner_bounds[:4]
    )
    buf_lo, buf_hi = k_layers1[0] * 3 + 2, (k_layers1[-1] + 1) * 3 - 1 - 2
    k_layers2 = [
        k
        for k in k_range_from_z(inner_bounds[4], inner_bounds[5], ZMIN, hz / 3, 3 * NZ)
        if buf_lo <= k <= buf_hi
    ]
    f_domain, f_outer, f_inner = fractions
    return (
        f_domain * NX * NY * NZ
        + 26 * f_outer * len(footprint1) * len(k_layers1)
        + 26 * f_inner * len(footprint2) * len(k_layers2)
    )


def build_double_refined_grid(
    cell_edge,
    XMIN,
    YMIN,
    ZMIN,
    Lx_dtm,
    Ly_dtm,
    Lz,
    inner_bounds,
    outer_bounds,
    terrain,
    validate_mesh=False,
):
    """One full build attempt at the given (isotropic-xy) cell_edge: grid
    sizing, both nested regions' footprints/k-layers, level-1 build_mesh,
    level-2 refine_region_further, and carving. Returns (points, hexes,
    tags, keep, grid_info) -- grid_info is a dict of everything printed/
    reused by the caller (NX, NY, NZ, h_xy, hz, XMAX, YMAX, footprint
    sizes, conformity results). Set validate_mesh=True to run the exhaustive
    face-based conformity checks; they are disabled by default for production
    meshes because they require storing and sorting six faces per hex.

    general_rebuild's point-dedup rounds position/cell_size to 5 decimal
    places (see build_mesh's gid() docstring) -- fine for the small,
    origin-relative offsets its own tests use, but a real DTM's
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
    LX0 = LY0 = LZ0 = (
        0.0  # local build frame; translated to (XMIN,YMIN,ZMIN) at the end
    )

    (inner_xmin, inner_xmax, inner_ymin, inner_ymax, inner_zbot, inner_ztop) = (
        inner_bounds
    )
    (outer_xmin, outer_xmax, outer_ymin, outer_ymax, outer_zbot, outer_ztop) = (
        outer_bounds
    )

    footprint1 = footprint_from_region(
        XMIN, YMIN, h_xy, h_xy, NX, NY, outer_xmin, outer_xmax, outer_ymin, outer_ymax
    )
    if not footprint1:
        raise ValueError("outer region does not overlap any coarse cell center")
    k_layers1 = k_range_from_z(outer_zbot, outer_ztop, ZMIN, hz, NZ)

    points, hexes, tags = GR.build_mesh(
        footprint1,
        NX=NX,
        NY=NY,
        NZ=NZ,
        XMIN=LX0,
        YMIN=LY0,
        ZMIN=LZ0,
        k_layers=k_layers1,
        cell_size=(h_xy, h_xy, hz),
    )
    print(
        f"[level 1] {len(footprint1)} footprint cells x {len(k_layers1)} layers -> "
        f"{len(hexes):,} hexes, peak memory {peak_memory_gb():.2f} GB",
        flush=True,
    )
    if validate_mesh:
        ok1, detail1 = GR.verify_mesh(
            points,
            hexes,
            footprint1,
            NX,
            NY,
            LX0,
            NX * h_xy,
            LY0,
            NY * h_xy,
            LZ0,
            NZ * hz,
            n_layers=len(k_layers1),
        )
        if not ok1:
            raise RuntimeError(f"level-1 refined mesh is not conforming: {detail1}")
    else:
        detail1 = "skipped (pass validate_mesh=True to run the exhaustive check)"

    # level 2: the inner region, one scale finer, nested inside the
    # level-1 urz block (strictly inside the outer region by construction).
    NXf, NYf, NZf = NX * 3, NY * 3, NZ * 3
    hxf, hyf, hzf = h_xy / 3, h_xy / 3, hz / 3
    footprint2 = footprint_from_region(
        XMIN, YMIN, hxf, hyf, NXf, NYf, inner_xmin, inner_xmax, inner_ymin, inner_ymax
    )
    if not footprint2:
        raise ValueError("inner region does not overlap any level-1-child cell center")
    buf_lo = (
        k_layers1[0] * 3 + 2
    )  # +2: mandatory same-scale buffer (local_size=5, buf=2)
    buf_hi = (k_layers1[-1] + 1) * 3 - 1 - 2
    k_layers2 = [
        k
        for k in k_range_from_z(inner_zbot, inner_ztop, ZMIN, hzf, NZf)
        if buf_lo <= k <= buf_hi
    ]
    if not k_layers2:
        raise ValueError(
            "inner region's z range leaves no room for the level-2 vertical buffer"
        )

    points, hexes, tags = GR.refine_region_further(
        points,
        hexes,
        tags,
        footprint2,
        k_layers2,
        NXf,
        NYf,
        NZf,
        LX0,
        LY0,
        LZ0,
        cell_size=(hxf, hyf, hzf),
    )
    print(
        f"[level 2] {len(footprint2)} footprint cells x {len(k_layers2)} layers -> "
        f"{len(hexes):,} hexes, peak memory {peak_memory_gb():.2f} GB",
        flush=True,
    )
    validation_t0 = time.perf_counter()
    if validate_mesh:
        ok2 = check_conformity_3d(
            points, hexes, LX0, NX * h_xy, LY0, NY * h_xy, LZ0, NZ * hz
        )
        if not ok2:
            raise RuntimeError("double-refined mesh is not conforming")
    else:
        print(
            "[conformity] exhaustive level-2 check skipped; continuing to carving",
            flush=True,
        )

    carve_t0 = time.perf_counter()
    print(
        f"[carve] starting with {len(hexes):,} hexes and {len(points):,} points; "
        f"mesh construction phase took {carve_t0 - validation_t0:.1f}s",
        flush=True,
    )
    points = points + np.array([XMIN, YMIN, ZMIN])
    print(
        f"[carve] translated points in {time.perf_counter() - carve_t0:.1f}s; "
        "materializing hex corner coordinates",
        flush=True,
    )

    # carve: drop every hex that is not in the rock
    coords = points[hexes]  # (n_hex, 8, 3)
    print(
        f"[carve] corner coordinates ready in {time.perf_counter() - carve_t0:.1f}s; "
        f"array size {coords.nbytes / 1024**3:.2f} GiB; asking the terrain",
        flush=True,
    )
    keep = terrain.keep(coords)
    print(
        f"[carve] keep mask ready in {time.perf_counter() - carve_t0:.1f}s; "
        f"keeping {int(keep.sum()):,} / {len(keep):,} hexes; peak memory {peak_memory_gb():.2f} GB",
        flush=True,
    )

    grid_info = dict(
        NX=NX,
        NY=NY,
        NZ=NZ,
        h_xy=h_xy,
        hz=hz,
        XMAX=XMAX,
        YMAX=YMAX,
        n_footprint1=len(footprint1),
        k_layers1=k_layers1,
        n_footprint2=len(footprint2),
        k_layers2=k_layers2,
        detail1=detail1,
    )
    return points, hexes, tags, keep, grid_info


def run(case):
    """Build the refined carved grid of the case's DTM -> grid.vtu, frame.npz,
    dtm_surface.stl, all in the grid's frame."""
    terrain, rotation, center = terrain_surfaces.build(case)
    print(f"terrain ready, peak memory {peak_memory_gb():.2f} GB", flush=True)
    case.output.mkdir(parents=True, exist_ok=True)
    frame.save(case.frame_path, rotation, center)
    print(f"wrote {case.frame_path}", flush=True)

    XMIN, YMIN = terrain.xmin, terrain.ymin
    ZMIN = terrain.zmin - case.z_padding
    ZMAX = terrain.zmax + case.z_padding
    Lz = ZMAX - ZMIN
    Lx_dtm = terrain.xmax - XMIN
    Ly_dtm = terrain.ymax - YMIN
    print(
        f"domain x=[{XMIN:.2f},{terrain.xmax:.2f}] y=[{YMIN:.2f},{terrain.ymax:.2f}] "
        f"z=[{ZMIN:.2f},{ZMAX:.2f}] (terrain z [{terrain.zmin:.2f},{terrain.zmax:.2f}] "
        f"+ padding {case.z_padding})",
        flush=True,
    )

    kept_fraction = terrain.kept_fraction(terrain.xmin, terrain.xmax, terrain.ymin, terrain.ymax, ZMIN, Lz)
    print(
        f"estimated below-terrain (kept) fraction of the box: {kept_fraction:.4f}, "
        f"peak memory {peak_memory_gb():.2f} GB",
        flush=True,
    )

    # the two nested regions' physical bounds don't depend on grid
    # resolution -- computed once, outside the cell_edge calibration loop.
    inner_xmin, inner_xmax, inner_ymin, inner_ymax = case.inner_region
    inner_zbot, inner_ztop = terrain.z_extent(
        inner_xmin, inner_xmax, inner_ymin, inner_ymax, case.region_z_padding
    )
    inner_bounds = (
        inner_xmin,
        inner_xmax,
        inner_ymin,
        inner_ymax,
        inner_zbot,
        inner_ztop,
    )
    scaled_outer = (
        *scale_interval(inner_xmin, inner_xmax, case.outer_scale),
        *scale_interval(inner_ymin, inner_ymax, case.outer_scale),
        *scale_interval(inner_zbot, inner_ztop, case.outer_scale),
    )
    print(
        f"inner region: x=[{inner_xmin:.1f},{inner_xmax:.1f}] "
        f"y=[{inner_ymin:.1f},{inner_ymax:.1f}] z=[{inner_zbot:.1f},{inner_ztop:.1f}]"
    )
    print(
        f"outer region ({case.outer_scale}x, same center, widened per build to 1.5 "
        f"coarse cells of margin): x=[{scaled_outer[0]:.1f},{scaled_outer[1]:.1f}] "
        f"y=[{scaled_outer[2]:.1f},{scaled_outer[3]:.1f}] "
        f"z=[{scaled_outer[4]:.1f},{scaled_outer[5]:.1f}]"
    )

    # size without building: start from the plain-grid estimate and rescale
    # until the predicted carved count, refinement included, meets the target
    fractions = (
        kept_fraction,
        terrain.kept_fraction(*scaled_outer[:4], scaled_outer[4], scaled_outer[5] - scaled_outer[4]),
        terrain.kept_fraction(*inner_bounds[:4], inner_bounds[4], inner_bounds[5] - inner_bounds[4]),
    )
    cell_edge = (Lx_dtm * Ly_dtm * Lz * kept_fraction / case.target_cells) ** (
        1.0 / 3.0
    )
    predicted = predicted_cells(
        cell_edge, XMIN, YMIN, ZMIN, Lx_dtm, Ly_dtm, Lz, inner_bounds, case.outer_scale, fractions
    )
    print(
        f"[presize] rock fractions: domain {fractions[0]:.3f}, outer {fractions[1]:.3f}, "
        f"inner {fractions[2]:.3f}; plain-grid cell_edge {cell_edge:.3f} would give "
        f"{predicted:,.0f} carved cells ({predicted / case.target_cells:.1f}x target)",
        flush=True,
    )
    for _ in range(PRESIZE_ITERATIONS):
        if predicted <= 0 or abs(predicted / case.target_cells - 1.0) < 0.02:
            break
        cell_edge *= (predicted / case.target_cells) ** (1.0 / 3.0)
        predicted = predicted_cells(
            cell_edge, XMIN, YMIN, ZMIN, Lx_dtm, Ly_dtm, Lz, inner_bounds, case.outer_scale, fractions
        )
    print(
        f"[presize] cell_edge {cell_edge:.3f} -> predicted {predicted:,.0f} carved cells "
        f"(target {case.target_cells:,})",
        flush=True,
    )
    for iteration in range(MAX_ITERATIONS):
        outer_bounds = outer_region(inner_bounds, case.outer_scale, cell_edge)
        print(
            f"[iter {iteration}] outer region at cell_edge {cell_edge:.3f}: "
            f"x=[{outer_bounds[0]:.1f},{outer_bounds[1]:.1f}] y=[{outer_bounds[2]:.1f},{outer_bounds[3]:.1f}] "
            f"z=[{outer_bounds[4]:.1f},{outer_bounds[5]:.1f}]",
            flush=True,
        )
        points, hexes, tags, keep, grid = build_double_refined_grid(
            cell_edge,
            XMIN,
            YMIN,
            ZMIN,
            Lx_dtm,
            Ly_dtm,
            Lz,
            inner_bounds,
            outer_bounds,
            terrain,
            validate_mesh=case.validate_mesh,
        )
        n_kept = int(keep.sum())
        ratio = n_kept / case.target_cells
        print(
            f"[iter {iteration}] cell_edge={cell_edge:.3f} -> grid "
            f"{grid['NX']}x{grid['NY']}x{grid['NZ']}, level 1 "
            f"{grid['n_footprint1']} cells / level 2 {grid['n_footprint2']} cells, "
            f"{len(hexes):,} hexes, {n_kept:,} kept (target {case.target_cells:,}, "
            f"ratio {ratio:.2f}), peak memory {peak_memory_gb():.2f} GB",
            flush=True,
        )
        if abs(ratio - 1.0) <= SIZE_TOLERANCE or iteration == MAX_ITERATIONS - 1:
            break
        cell_edge *= ratio ** (
            1.0 / 3.0
        )  # refinement inflates cell count beyond the plain-grid estimate

    if case.validate_mesh:
        print(f"level 1 conformity check: {grid['detail1']}")
        print("level 2 (combined) conformity check: PASS")
    else:
        print("conformity checks: skipped (validate_mesh=False)")

    kept_hexes = hexes[keep]
    kind_str = [t.split(":")[0] for t, k in zip(tags, keep) if k]
    kind_names = sorted(set(kind_str))
    kind_code = np.array([kind_names.index(k) for k in kind_str], dtype=float)

    terrain.export(case.dtm_surface_path)
    meshio.write_points_cells(
        case.grid_path,
        points,
        [("hexahedron", kept_hexes)],
        cell_data={"kind_code": [kind_code]},
    )
    print(f"wrote {case.grid_path} -- kind_code legend: {list(enumerate(kind_names))}")
