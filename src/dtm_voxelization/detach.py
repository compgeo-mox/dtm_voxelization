"""Split the grid along the cut faces of ALL the case's surfaces at once.

The cells stay exactly where they are, but the two sides of every crack stop
sharing points, so they are logically disconnected -- cracks with zero
opening. Doing all surfaces in one split is what makes crossing or touching
fractures come out right: a node on the line where two cracks meet simply
gets one copy per group of cells the cracks separate.

The split is per node, not per face. For each node on a crack, walk the cells
around it, hopping only through faces that are NOT cut faces: each group of
cells that can still reach each other gets its own copy of the node. Groups
on opposite sides of a crack therefore end up with distinct (but
coincident) points, while a node at a crack TIP -- where the cells wrap
around the end of the surface and remain mutually reachable -- stays a
single point, which keeps the crack closed at its tip instead of tearing the
mesh open to the boundary.

Checked, failing loudly otherwise:

- no two cells across a cut face still share a node away from a crack's rim
  (its tip), checked on every cut face;
- a sample of non-cut neighbour pairs still share all 4 face nodes.

Fields in `detached.vtu`, for a look in ParaView:

- `crack_cell` -- 1 for cells touching a crack;
- `side_color` -- those cells coloured by what they can still reach through
  shared nodes, ignoring the nodes welded across a crack at its tip: the sides
  of a crack get different colours, unless they meet around another crack's
  end;
- `opening` -- point displacement pulling every crack apart; *Warp By
  Vector* on it opens the cracks, which is only possible because their
  nodes are now distinct points, and the tips stay stitched.
"""

import numpy as np
import meshio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geometry_checks import hex_volumes_signed
from .cut import crossings_with_jitter
from .mesh import build_face_table, read_hex_mesh, require
from .surfaces import in_grid_frame

OPENING = 0.35  # warp amplitude, as a fraction of the local cut-face size
PROBE_PAIRS = 200_000  # (point, triangle) pairs probed at once, in `beyond`


def find_cut_faces(table_faces, cut_faces):
    """Mask of the cut faces inside the mesh's own face table."""
    table_keys, wanted = np.sort(table_faces, axis=1), np.sort(cut_faces, axis=1)
    _, inverse = np.unique(np.vstack([table_keys, wanted]), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    table_ids, wanted_ids = inverse[: len(table_keys)], inverse[len(table_keys) :]

    mask = np.isin(table_ids, wanted_ids)
    if int(mask.sum()) != len(np.unique(wanted_ids)):
        raise ValueError(
            "some cut faces are not faces of this grid -- rerun 'cut' after 'grid'"
        )
    return mask


def node_cell_incidences(hexes, nodes):
    """Every (node, cell) pair for the given nodes, with a sortable key."""
    on_crack = np.isin(hexes, nodes)
    cells, corners = np.nonzero(on_crack)
    node_ids = hexes[cells, corners]

    keys = node_ids.astype(np.int64) * len(hexes) + cells
    unique_keys, first = np.unique(keys, return_index=True)
    return node_ids[first], cells[first], unique_keys


def split_nodes(points, hexes, faces, owners, cut_mask):
    """Duplicate crack nodes so the sides of each crack stop sharing them.

    Returns the new points and connectivity plus the incidence table
    (node, cell, new node id) used to build the opening field."""
    crack_nodes = np.unique(faces[cut_mask])
    inc_node, inc_cell, inc_keys = node_cell_incidences(hexes, crack_nodes)

    # cells that can still reach each other around a node -- every internal
    # face that is not cut welds its two cells together at each of its nodes
    internal = owners[:, 1] >= 0
    weld = internal & ~cut_mask
    weld_faces, weld_owners = faces[weld], owners[weld]
    touches = np.isin(weld_faces, crack_nodes)
    fi, fj = np.nonzero(touches)
    edge_node = weld_faces[fi, fj]
    left = np.searchsorted(
        inc_keys, edge_node.astype(np.int64) * len(hexes) + weld_owners[fi, 0]
    )
    right = np.searchsorted(
        inc_keys, edge_node.astype(np.int64) * len(hexes) + weld_owners[fi, 1]
    )

    n_inc = len(inc_keys)
    graph = coo_matrix((np.ones(len(left)), (left, right)), shape=(n_inc, n_inc))
    _, group = connected_components(graph, directed=False)

    # one copy of the node per group; the first group keeps the original id
    pairs = np.stack([inc_node, group], axis=1)
    _, copy_id = np.unique(pairs, axis=0, return_inverse=True)
    copy_id = copy_id.ravel()
    first_copy = np.zeros(copy_id.max() + 1, dtype=bool)
    order = np.lexsort((copy_id, inc_node))
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = inc_node[order][1:] != inc_node[order][:-1]
    first_copy[copy_id[order][keep]] = True

    new_id = np.empty(copy_id.max() + 1, dtype=np.int64)
    n_new = int((~first_copy).sum())
    new_id[~first_copy] = len(points) + np.arange(n_new)
    original_of_copy = np.zeros(copy_id.max() + 1, dtype=np.int64)
    original_of_copy[copy_id] = inc_node
    new_id[first_copy] = original_of_copy[first_copy]

    inc_new = new_id[copy_id]
    hexes_out = hexes.copy()
    hexes_out[inc_cell, np.argmax(hexes[inc_cell] == inc_node[:, None], axis=1)] = inc_new
    points_out = np.vstack([points, points[original_of_copy[~first_copy]]])

    return points_out, hexes_out, dict(
        new_node=inc_new,
        keys=inc_keys,
        n_duplicated=n_new,
        n_crack_nodes=len(crack_nodes),
    )


def rim_nodes(cut_faces):
    """Nodes on the rim of the staircase surfaces -- their tip lines.

    An edge used by a single cut face bounds a sheet; the crack closes along
    it, so the cells there legitimately keep sharing those nodes."""
    edges = np.sort(
        np.concatenate([cut_faces[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique_edges[counts == 1])


def face_copies(hexes, faces, owners, cut_mask, inc):
    """The copy of each of a cut face's 4 nodes used by the cell on either
    side: two (F, 4) arrays of new node ids. Where they coincide the node is
    still welded across the crack."""
    keys = faces[cut_mask].astype(np.int64) * len(hexes)
    return tuple(
        inc["new_node"][np.searchsorted(inc["keys"], keys + owners[cut_mask][:, [side]])]
        for side in (0, 1)
    )


def color_sides(new_hexes, touched, welded):
    """Colour the cells around the cracks by what they can still reach.

    Two touched cells are joined when they share a node that is not welded
    across a crack -- the welds sit at the tips, where a crack is closed."""
    node_lists = new_hexes[touched]
    rows, cols = np.nonzero(~np.isin(node_lists, welded))
    _, node_index = np.unique(node_lists[rows, cols], return_inverse=True)
    node_index = node_index.ravel()

    n_cells, n_nodes = len(touched), node_index.max() + 1
    graph = coo_matrix(
        (np.ones(len(rows)), (rows, n_cells + node_index)),
        shape=(n_cells + n_nodes, n_cells + n_nodes),
    )
    _, labels = connected_components(graph, directed=False)
    _, colors = np.unique(labels[:n_cells], return_inverse=True)
    return colors.ravel() + 1


def build_opening(points, hexes, faces, owners, cut_mask, inc):
    """Displacement moving each copy of a crack node away from the cells
    across the crack; at a tip the two pulls cancel and the node stays put."""
    centers = points[hexes].mean(axis=1)
    cut_faces, cut_owners = faces[cut_mask], owners[cut_mask]

    accum = np.zeros((len(points) + inc["n_duplicated"], 3))
    weight = np.zeros(len(accum))
    for own, other in ((0, 1), (1, 0)):
        away = centers[cut_owners[:, own]] - centers[cut_owners[:, other]]
        away /= np.linalg.norm(away, axis=1, keepdims=True)
        cells = np.repeat(cut_owners[:, own], 4)
        keys = cut_faces.ravel().astype(np.int64) * len(hexes) + cells
        copy = inc["new_node"][np.searchsorted(inc["keys"], keys)]
        np.add.at(accum, copy, np.repeat(away, 4, axis=0))
        np.add.at(weight, copy, 1.0)

    moving = weight > 0
    accum[moving] /= weight[moving][:, None]
    scale = np.median(
        np.linalg.norm(points[cut_faces[:, 1]] - points[cut_faces[:, 0]], axis=1)
    )
    return accum * OPENING * scale


def check_split(new_hexes, faces, owners, cut_mask, copies):
    """Cells across a cut face stay welded only at rim nodes; intact
    neighbours still share their 4 face nodes. Raises otherwise."""
    welded = copies[0] == copies[1]
    on_rim = np.isin(faces[cut_mask], rim_nodes(faces[cut_mask]))
    off_rim = (welded & ~on_rim).any(axis=1)
    print(
        f"  cut pairs {len(welded):,}: fully detached {int((~welded.any(axis=1)).sum()):,}, "
        f"welded along a tip rim {int((welded.any(axis=1) & ~off_rim).sum()):,}, "
        f"welded off the rim {int(off_rim.sum()):,}",
        flush=True,
    )
    if off_rim.any():
        raise RuntimeError(f"{int(off_rim.sum())} cut pairs still share nodes off the rim")

    rng = np.random.default_rng(0)
    intact = owners[(owners[:, 1] >= 0) & ~cut_mask]
    sample = intact[rng.choice(len(intact), size=min(2000, len(intact)), replace=False)]
    n_shared = np.array([len(np.intersect1d(new_hexes[a], new_hexes[b])) for a, b in sample])
    print(
        f"  intact neighbour pairs sampled {len(sample):,}: "
        f"sharing 4 nodes {int((n_shared == 4).sum()):,}",
        flush=True,
    )
    if (n_shared != 4).any():
        raise RuntimeError(f"{int((n_shared != 4).sum())} intact neighbour pairs came apart")


def components(hexes, n_points):
    """Which piece of the mesh each cell belongs to, cells sharing a node being
    of one piece. Walked as a bipartite cell-node graph rather than a cell-cell
    one: the adjacency product costs an order of magnitude more at real sizes,
    for the same answer."""
    n_cells = len(hexes)
    rows = np.repeat(np.arange(n_cells), hexes.shape[1])
    graph = coo_matrix(
        (np.ones(hexes.size, dtype=np.int8), (rows, hexes.ravel() + n_cells)),
        shape=(n_cells + n_points, n_cells + n_points),
    )
    return connected_components(graph, directed=False)[1][:n_cells]


def beyond(points, tri, rng):
    """Which points lie past the surface, on the side its normals face.

    A ray leaving the point that way crosses a finite surface an odd number of
    times exactly when the surface is in front of it, which is the same parity
    argument the carving and the cut both rest on. Only the points the surface
    can shadow are probed -- the rest cannot cross it at all."""
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(axis=0)
    normal /= np.linalg.norm(normal)
    span = float(np.linalg.norm(np.ptp(tri.reshape(-1, 3), axis=0)))

    flat = tri.reshape(-1, 3)
    middle = flat.mean(axis=0)
    # full_matrices=False, or LAPACK builds a U of (3 x triangles) squared
    axes = np.linalg.svd(flat - middle, full_matrices=False)[2][:2]  # in-plane axes
    shadow = (flat - middle) @ axes.T
    projected = (points - middle) @ axes.T
    candidate = np.all(
        (projected >= shadow.min(axis=0)) & (projected <= shadow.max(axis=0)), axis=1
    )

    # in batches: count_crossings holds ten (pairs, 3) arrays of doubles at a
    # time, and its own 2M-pair budget costs gigabytes on a surface of a few
    # thousand triangles. Nothing here is in a hurry.
    out = np.zeros(len(points), dtype=bool)
    idx = np.flatnonzero(candidate)
    stride = max(1, PROBE_PAIRS // max(1, len(tri)))
    for lo in range(0, len(idx), stride):
        here = idx[lo : lo + stride]
        start = points[here]
        counts = crossings_with_jitter(start, start + 3 * span * normal, tri, rng)
        out[here] = counts % 2 == 1
    return out, int(candidate.sum())


def prune(new_points, new_hexes, owners, masks, case, crack_cell, side_color):
    """Drop what the surfaces marked `remove_detached` cut loose: every cell
    with even one corner past the surface, on the side its normals face.

    The cells the split leaves in a piece of their own go too, by construction,
    and their count is reported against the geometric one: a surface that cuts
    a block free should cover it, and a large gap between the two means the
    surface faces the wrong way. Returns (hexes, cell arrays, keep) of the
    survivors. The points are left alone, unused ones and all: grid.vtu already
    carries the points of every hex the carving dropped, and renumbering here
    would throw those away too -- a change to make deliberately or not at
    all."""
    keep = np.ones(len(new_hexes), dtype=bool)
    wanted = [s for s in case.surfaces if s.remove_detached]
    if not wanted:
        return new_hexes, crack_cell, side_color, keep

    rng = np.random.default_rng(0)
    piece = components(new_hexes, len(new_points))
    sizes = np.bincount(piece)
    print(f"  the split mesh is in {len(sizes)} piece(s), the largest {sizes.max():,} cells")
    for surface in wanted:
        tri = in_grid_frame(case, surface)
        past, probed = beyond(new_points, tri, rng)
        doomed = past[new_hexes].any(axis=1)
        loose = np.unique(piece[np.unique(owners[masks[surface.name]])])
        loose = loose[sizes[loose] < sizes[loose].max()]
        cut_free = np.isin(piece, loose)
        print(
            f"  {surface.name}: {probed:,} of {len(new_points):,} points stand in its shadow, "
            f"{int(past.sum()):,} of them past it -> {int(doomed.sum()):,} cells with a corner "
            f"beyond, against {int(cut_free.sum()):,} the split cut loose",
            flush=True,
        )
        if cut_free.any() and not doomed[cut_free].all():
            print(
                f"  WARNING: {int((~doomed[cut_free]).sum()):,} of the cells it cut loose are "
                f"NOT past it -- the surface may face the wrong way, check OUTWARD",
                flush=True,
            )
        keep &= ~(doomed | cut_free)

    print(f"  {int((~keep).sum()):,} cells removed", flush=True)
    return new_hexes[keep], crack_cell[keep], side_color[keep], keep


def run(case):
    """Split grid.vtu along the cut faces of every surface -> detached.vtu."""
    require(case.grid_path, "grid")
    points, hexes = read_hex_mesh(case.grid_path)
    faces, owners = build_face_table(hexes)
    print(f"  grid {case.grid_path}: {len(hexes):,} hexes, {len(points):,} points", flush=True)

    cut_mask = np.zeros(len(faces), dtype=bool)
    masks = {}
    for surface in case.surfaces:
        path = case.cut_dir / f"{surface.name}.npz"
        require(path, "cut")
        mask = masks[surface.name] = find_cut_faces(faces, np.load(path)["face_nodes"])
        print(
            f"  {surface.name}: {int(mask.sum()):,} cut faces, "
            f"{int((mask & cut_mask).sum()):,} of them already cut by an earlier surface",
            flush=True,
        )
        cut_mask |= mask
    n_cut = int(cut_mask.sum())
    print(f"  {n_cut:,} cut faces in total", flush=True)

    crack_cell = np.zeros(len(hexes))
    side_color = np.zeros(len(hexes))
    if n_cut == 0:
        np.savez(case.detach_summary_path, twin_pairs=0)
        print("  nothing to split, writing the grid unchanged", flush=True)
        new_points, new_hexes, opening = points, hexes, np.zeros_like(points)
    else:
        new_points, new_hexes, inc = split_nodes(points, hexes, faces, owners, cut_mask)
        print(
            f"  {inc['n_crack_nodes']:,} crack nodes -> {inc['n_duplicated']:,} "
            f"duplicated ({len(new_points):,} points now)",
            flush=True,
        )
        copies = face_copies(hexes, faces, owners, cut_mask, inc)
        check_split(new_hexes, faces, owners, cut_mask, copies)

        touched = np.unique(owners[cut_mask])
        welded = np.unique(copies[0][copies[0] == copies[1]])
        colors = color_sides(new_hexes, touched, welded)
        crack_cell[touched] = 1.0
        side_color[touched] = colors
        cut_owners = owners[cut_mask]
        same_color = int((side_color[cut_owners[:, 0]] == side_color[cut_owners[:, 1]]).sum())
        print(
            f"  {len(np.unique(colors))} colours on {len(touched):,} cells touching a crack "
            f"({', '.join(str(int((colors == c).sum())) for c in np.unique(colors))} cells); "
            f"cut faces with the same colour on both sides: {same_color:,}",
            flush=True,
        )
        opening = build_opening(points, hexes, faces, owners, cut_mask, inc)
        new_hexes, crack_cell, side_color, keep = prune(
            new_points, new_hexes, owners, masks, case, crack_cell, side_color
        )

        # what speed should find: a cut face becomes a pair of boundary faces
        # only if it opened -- one sealed at a tip stays internal -- and only if
        # both the cells it separates are still here
        sealed = (copies[0] == copies[1]).all(axis=1)
        cut_owners = owners[cut_mask]
        both_here = keep[cut_owners[:, 0]] & keep[cut_owners[:, 1]]
        twin_pairs = int((~sealed & both_here).sum())
        np.savez(case.detach_summary_path, twin_pairs=twin_pairs)
        print(
            f"  of {n_cut:,} cut faces, {int(sealed.sum()):,} stayed sealed at a tip and "
            f"{int((~both_here).sum()):,} lost a side to the pruning: {twin_pairs:,} twin pairs",
            flush=True,
        )

    case.output.mkdir(parents=True, exist_ok=True)
    meshio.write_points_cells(
        case.detached_path,
        new_points,
        [("hexahedron", new_hexes)],
        cell_data={
            "crack_cell": [crack_cell],
            "side_color": [side_color],
            "volume": [np.abs(hex_volumes_signed(new_points, new_hexes))],
        },
        point_data={"opening": opening},
    )
    print(f"  wrote {case.detached_path}", flush=True)
