"""Voxelize each fracture surface onto the grid: the staircase set of grid
faces the surface cuts through.

The voxelized image is a set of *existing* grid faces -- no cell is cut, no
node is moved -- chosen so that it separates the cells on one side of the
sheet from the cells on the other.

Selection rule: a face belongs to the cut set when the segment joining the
two cells sharing it (its *dual edge*) crosses the surface an ODD number of
times. Parity is what makes the result a continuous sheet rather than a
fuzzy band: walk from any cell to any other and the number of cut faces you
traverse has the same parity as the number of times the path pierces the
surface, so the cut faces close up into a staircase surface with no
pinholes. It is also exactly the set the detachment step needs -- each cut
face is one place where the two cells must stop sharing nodes.

Ties are broken consistently. A surface lying exactly on the cell centres
(a flat sheet on a Cartesian grid is the normal case, not an exotic one)
leaves every probe undecided, and deciding each one independently shreds
the sheet -- so the SURFACE is nudged once, by a fixed sub-micron offset
along a direction skew to the grid, which moves every probe to the same
side of the tie. The dual edge is a probe, not geometry, so what is left
after that -- a probe through a triangle edge, shared by two triangles and
counted twice -- is re-thrown with a small random jitter until its parity
is unambiguous.

Open sheets are fine: the surface may terminate inside the grid, and the
cut set then simply stops there. Faces on the grid boundary -- including
the ones the terrain carving exposed -- have only one owner and no dual
edge, so a sheet sticking out of the grid contributes nothing there.

Writes, per surface, `cut_faces/<name>.npz` (`face_nodes` (F, 4),
`cell_pairs` (F, 2), in the grid's numbering) and, when not empty,
`cut_faces/<name>.vtu` for a look in ParaView.
"""

import numpy as np
import meshio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mesh import build_face_table, read_hex_mesh, require
from .surfaces import in_grid_frame

MAX_JITTER_RETRIES = 8  # re-throws allowed for an ambiguous dual edge
TIE_BREAK = 1e-6  # global surface nudge, in units of the local cell size
# skew to every grid axis and face diagonal, so the nudge never re-creates
# the tie it is there to break
TIE_BREAK_DIR = np.array([0.3178, 0.4571, 0.8309])
CHUNK = 2_000_000  # max (segment, triangle) pairs tested at once


def count_crossings(seg_start, seg_end, tri, eps=1e-9, ambiguity_tol=1e-7):
    """Moller-Trumbore, vectorized over segments x triangles.

    Returns (counts, ambiguous): how many triangles each segment pierces,
    and which segments had a near-miss -- a hit landing on a triangle edge
    (counted twice, once per neighbouring triangle) or at a segment
    endpoint -- where the parity cannot be trusted."""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    edge1, edge2 = v1 - v0, v2 - v0

    counts = np.zeros(len(seg_start), dtype=np.int64)
    ambiguous = np.zeros(len(seg_start), dtype=bool)

    stride = max(1, CHUNK // max(1, len(tri)))
    for lo in range(0, len(seg_start), stride):
        hi = min(lo + stride, len(seg_start))
        origin = seg_start[lo:hi, None, :]
        direction = (seg_end[lo:hi] - seg_start[lo:hi])[:, None, :]

        pvec = np.cross(direction, edge2[None])
        det = np.einsum("...i,...i->...", edge1[None], pvec)
        parallel = np.abs(det) < eps
        inv_det = 1.0 / np.where(parallel, 1.0, det)

        tvec = origin - v0[None]
        u = np.einsum("...i,...i->...", tvec, pvec) * inv_det
        qvec = np.cross(tvec, edge1[None])
        v = np.einsum("...i,...i->...", direction, qvec) * inv_det
        t = np.einsum("...i,...i->...", edge2[None], qvec) * inv_det

        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        hit = ~parallel & inside & (t > 0.0) & (t < 1.0)
        counts[lo:hi] = hit.sum(axis=1)

        # a hit whose barycentric or along-segment coordinate sits on a
        # boundary is shared with a neighbouring triangle, or with the cell
        # centre itself -- parity there is a coin flip
        near_border = (
            (np.abs(u) < ambiguity_tol)
            | (np.abs(v) < ambiguity_tol)
            | (np.abs(u + v - 1.0) < ambiguity_tol)
            | (np.abs(t) < ambiguity_tol)
            | (np.abs(t - 1.0) < ambiguity_tol)
        )
        plausible = ~parallel & (u > -ambiguity_tol) & (v > -ambiguity_tol)
        plausible &= (u + v < 1.0 + ambiguity_tol) & (t > -ambiguity_tol)
        plausible &= t < 1.0 + ambiguity_tol
        ambiguous[lo:hi] = (near_border & plausible).any(axis=1)

    return counts, ambiguous


def crossings_with_jitter(seg_start, seg_end, tri, rng, max_retries=MAX_JITTER_RETRIES):
    """`count_crossings`, re-throwing ambiguous probes with a tiny random
    offset until their parity is unambiguous."""
    counts, ambiguous = count_crossings(seg_start, seg_end, tri)

    scale = np.linalg.norm(seg_end - seg_start, axis=1)
    for attempt in range(max_retries):
        idx = np.flatnonzero(ambiguous)
        if len(idx) == 0:
            break
        step = 1e-6 * (2.0**attempt) * scale[idx, None]
        offset = rng.normal(size=(len(idx), 3)) * step
        counts[idx], ambiguous[idx] = count_crossings(
            seg_start[idx] + offset, seg_end[idx] + offset, tri
        )

    n_left = int(ambiguous.sum())
    if n_left:
        print(f"  warning: {n_left} dual edges still ambiguous after jitter")
    return counts


def voxelize_surface(points, hexes, tri, face_table=None, seed=0, tie_break=TIE_BREAK):
    """Grid faces whose dual edge crosses the surface an odd number of times.

    Returns a dict with `face_nodes` (F, 4), `cell_pairs` (F, 2) and the
    number of candidates actually tested."""
    face_nodes, owners = face_table if face_table is not None else build_face_table(hexes)
    internal = owners[:, 1] >= 0
    int_faces, int_owners = face_nodes[internal], owners[internal]

    centers = points[hexes].mean(axis=1)
    seg_start = centers[int_owners[:, 0]]
    seg_end = centers[int_owners[:, 1]]

    # only dual edges whose bounding box meets the surface's can cross it
    lo = np.minimum(seg_start, seg_end)
    hi = np.maximum(seg_start, seg_end)
    pad = 1e-9 + 1e-6 * np.ptp(tri.reshape(-1, 3), axis=0)
    tri_lo = tri.reshape(-1, 3).min(axis=0) - pad
    tri_hi = tri.reshape(-1, 3).max(axis=0) + pad
    candidate = np.all((hi >= tri_lo) & (lo <= tri_hi), axis=1)
    cand_idx = np.flatnonzero(candidate)
    if len(cand_idx) == 0:
        return dict(face_nodes=int_faces[:0], cell_pairs=int_owners[:0], n_candidates=0)

    # one global tie-break: cell centres sitting exactly on the surface are
    # the rule for flat sheets on a Cartesian grid, and every such probe has
    # to fall on the same side or the staircase comes apart
    cell_size = np.median(
        np.linalg.norm(seg_end[cand_idx] - seg_start[cand_idx], axis=1)
    )
    direction = TIE_BREAK_DIR / np.linalg.norm(TIE_BREAK_DIR)
    tri = tri + tie_break * cell_size * direction

    counts = crossings_with_jitter(
        seg_start[cand_idx], seg_end[cand_idx], tri, np.random.default_rng(seed)
    )
    cut = cand_idx[counts % 2 == 1]

    return dict(
        face_nodes=int_faces[cut],
        cell_pairs=int_owners[cut],
        n_candidates=len(cand_idx),
    )


def count_connected_faces(face_nodes):
    """Number of edge-connected components of the staircase surface -- 1
    means the cut faces form a single continuous sheet."""
    if len(face_nodes) == 0:
        return 0
    edges = np.concatenate(
        [face_nodes[:, [i, (i + 1) % 4]] for i in range(4)], axis=0
    )
    _, edge_id = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    edge_id = edge_id.ravel()
    face_id = np.tile(np.arange(len(face_nodes)), 4)

    # faces and edges as one bipartite graph: faces sharing an edge land in
    # the same component
    n_faces, n_edges = len(face_nodes), edge_id.max() + 1
    graph = coo_matrix(
        (np.ones(len(face_id)), (face_id, n_faces + edge_id)),
        shape=(n_faces + n_edges, n_faces + n_edges),
    )
    _, labels = connected_components(graph, directed=False)
    return len(np.unique(labels[:n_faces]))


def explain_empty_cut(points, hexes, tri):
    """Why a surface cut nothing: outside the grid, inside carved-away
    space, or simply finer than the cells it lives in."""
    flat = tri.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    centers = points[hexes].mean(axis=1)
    sizes = np.ptp(points[hexes], axis=1)

    inside = np.all((centers >= lo) & (centers <= hi), axis=1)
    if inside.any():
        return f"{int(inside.sum())} cell centres inside its bounding box"

    near = np.all(
        (centers >= lo - 2 * sizes.max(axis=0)) & (centers <= hi + 2 * sizes.max(axis=0)),
        axis=1,
    )
    if not near.any():
        return (
            "no cells anywhere near it -- the surface sits outside the grid, "
            "or in space the terrain carving removed"
        )
    local = np.median(sizes[near], axis=0)
    span = hi - lo
    return (
        f"surface spans {span.round(2)} where the local cells are "
        f"{local.round(2)} -- it is smaller than one cell, so no dual edge "
        "can cross it"
    )


def run(case):
    """Voxelize every surface of the case onto grid.vtu -> cut_faces/<name>.npz."""
    if not case.surfaces:
        print("  no surfaces in the case", flush=True)
        return
    require(case.grid_path, "grid")
    points, hexes = read_hex_mesh(case.grid_path)
    print(f"  grid {case.grid_path}: {len(hexes):,} hexes, {len(points):,} points", flush=True)

    face_table = build_face_table(hexes)
    n_internal = int((face_table[1][:, 1] >= 0).sum())
    print(f"  {len(face_table[0]):,} faces ({n_internal:,} internal)", flush=True)

    case.cut_dir.mkdir(parents=True, exist_ok=True)
    for surface in case.surfaces:
        tri = in_grid_frame(case, surface)
        print(f"\n  {surface.name}: {len(tri)} triangles", flush=True)
        cut = voxelize_surface(points, hexes, tri, face_table=face_table)
        face_nodes, cell_pairs = cut["face_nodes"], cut["cell_pairs"]
        print(
            f"    {cut['n_candidates']:,} dual edges tested -> "
            f"{len(face_nodes):,} cut faces, "
            f"{len(np.unique(cell_pairs)):,} cells touched, "
            f"{count_connected_faces(face_nodes)} connected component(s)",
            flush=True,
        )

        npz_path = case.cut_dir / f"{surface.name}.npz"
        np.savez(npz_path, face_nodes=face_nodes, cell_pairs=cell_pairs)
        print(f"    wrote {npz_path}", flush=True)
        if len(face_nodes) == 0:
            print(f"    no faces cut: {explain_empty_cut(points, hexes, tri)}", flush=True)
            continue

        vtu_path = case.cut_dir / f"{surface.name}.vtu"
        meshio.write_points_cells(
            vtu_path,
            points,
            [("quad", face_nodes)],
            cell_data={
                "cell_left": [cell_pairs[:, 0].astype(float)],
                "cell_right": [cell_pairs[:, 1].astype(float)],
            },
        )
        print(f"    wrote {vtu_path}", flush=True)
