"""Detach the cells on the two sides of a voxelized cut surface.

Takes the staircase face set from `cut_surface.py` and splits the mesh
along it: the cells stay exactly where they are, but the two sides stop
sharing points, so they are logically disconnected -- a crack with zero
opening.

The split is per node, not per face. For each node on the crack, walk the
cells around it, hopping only through faces that are NOT cut faces: each
group of cells that can still reach each other gets its own copy of the
node. Two groups on opposite sides of the crack therefore end up with two
distinct (but coincident) points, while a node at the crack TIP -- where
the cells wrap around the end of the surface and remain mutually
reachable -- stays a single point, which is what keeps the crack closed at
its tip instead of tearing the mesh open to the boundary.

Verification, all reported by `main`:

- every cut face pair must share NO node afterwards (the detachment proper);
- every non-cut neighbour pair must still share its 4 nodes (nothing else
  came apart);
- each copy must sit exactly on top of the node it came from;
- the geometric sides, computed independently from the STL by nearest
  triangle, must disagree across every cut face.

Coloring, for a look in ParaView (`output/detached/<name>_detached.vtu`):

- `crack_side` -- cell data, -1 / +1 on the two sides of the surface
  (0 away from it), from the STL geometry rather than from the topology,
  so it is an independent check of what the splitting did;
- `crack_cell` -- cell data, 1 for cells touching the crack;
- `opening` -- point data, a displacement that pulls the two sides apart.
  Apply ParaView's *Warp By Vector* to it: the crack opens only because
  its nodes are now distinct points, so a visible gap IS the proof that
  the detachment worked, and the tip stays stitched.
"""

from pathlib import Path

import numpy as np
import meshio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# lets this file also run directly, not just as part of the package
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtm_voxelization.cut_surface import build_face_table

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_PATH = REPO_ROOT / "output" / "cartgrid_carved_refined.vtu"
CUT_DIR = REPO_ROOT / "output" / "cut_faces"
SURFACE_DIR = REPO_ROOT / "output" / "planes"
OUTPUT_DIR = REPO_ROOT / "output" / "detached"

OPENING = 0.35  # warp amplitude, as a fraction of the local cut-face size


def face_keys(face_nodes):
    """Order-independent key per face, for matching faces across arrays."""
    return np.sort(face_nodes, axis=1)


def find_cut_faces(table_faces, cut_faces):
    """Locate the cut faces inside the mesh's own face table."""
    table_keys, wanted = face_keys(table_faces), face_keys(cut_faces)
    _, inverse = np.unique(np.vstack([table_keys, wanted]), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    table_ids, wanted_ids = inverse[: len(table_keys)], inverse[len(table_keys) :]

    mask = np.isin(table_ids, wanted_ids)
    if int(mask.sum()) != len(np.unique(wanted_ids)):
        raise ValueError("some cut faces are not faces of this grid")
    return mask


def node_cell_incidences(hexes, nodes):
    """Every (node, cell) pair for the given nodes, plus a lookup helper."""
    on_crack = np.isin(hexes, nodes)
    cells, corners = np.nonzero(on_crack)
    node_ids = hexes[cells, corners]

    keys = node_ids.astype(np.int64) * len(hexes) + cells
    unique_keys, first = np.unique(keys, return_index=True)
    return node_ids[first], cells[first], unique_keys


def split_nodes(points, hexes, faces, owners, cut_mask):
    """Duplicate crack nodes so the two sides stop sharing them.

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
    graph = coo_matrix(
        (np.ones(len(left)), (left, right)), shape=(n_inc, n_inc)
    )
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
    new_id[first_copy] = -1  # filled below with the original node id
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
        node=inc_node,
        cell=inc_cell,
        new_node=inc_new,
        keys=inc_keys,
        origin_of_copy=original_of_copy[~first_copy],
        n_duplicated=n_new,
        n_crack_nodes=len(crack_nodes),
    )


def closest_point_on_triangles(query, tri, chunk=200_000):
    """Closest point on a triangle soup, and which triangle it is on."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    best_point = np.empty_like(query)
    best_tri = np.empty(len(query), dtype=np.int64)

    stride = max(1, chunk // max(1, len(tri)))
    for lo in range(0, len(query), stride):
        p = query[lo : lo + stride, None, :]
        ap = p - a[None]
        d1 = np.einsum("...i,...i->...", ab[None], ap)
        d2 = np.einsum("...i,...i->...", ac[None], ap)
        bp = p - b[None]
        d3 = np.einsum("...i,...i->...", ab[None], bp)
        d4 = np.einsum("...i,...i->...", ac[None], bp)
        cp = p - c[None]
        d5 = np.einsum("...i,...i->...", ab[None], cp)
        d6 = np.einsum("...i,...i->...", ac[None], cp)

        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        denom = 1.0 / np.where(va + vb + vc == 0, 1.0, va + vb + vc)
        v = np.clip(vb * denom, 0, 1)
        w = np.clip(vc * denom, 0, 1)

        # the six Voronoi regions of a triangle, in order
        on = a[None] + v[..., None] * ab[None] + w[..., None] * ac[None]
        on = np.where(
            ((d1 <= 0) & (d2 <= 0))[..., None], a[None] + 0 * on, on
        )
        on = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b[None] + 0 * on, on)
        on = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c[None] + 0 * on, on)
        t_ab = np.clip(np.where(d1 - d3 == 0, 0.0, d1 / np.where(d1 - d3 == 0, 1, d1 - d3)), 0, 1)
        on = np.where(
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
            a[None] + t_ab[..., None] * ab[None],
            on,
        )
        t_ac = np.clip(np.where(d2 - d6 == 0, 0.0, d2 / np.where(d2 - d6 == 0, 1, d2 - d6)), 0, 1)
        on = np.where(
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
            a[None] + t_ac[..., None] * ac[None],
            on,
        )
        denom_bc = (d4 - d3) + (d5 - d6)
        t_bc = np.clip(
            np.where(denom_bc == 0, 0.0, (d4 - d3) / np.where(denom_bc == 0, 1, denom_bc)),
            0,
            1,
        )
        on = np.where(
            ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None],
            b[None] + t_bc[..., None] * (c[None] - b[None]),
            on,
        )

        dist = np.einsum("...i,...i->...", p - on, p - on)
        pick = np.argmin(dist, axis=1)
        rows = np.arange(len(pick))
        best_point[lo : lo + stride] = on[rows, pick]
        best_tri[lo : lo + stride] = pick

    return best_point, best_tri


def geometric_side(query, tri, tol=1e-12):
    """Which side of the triangulated sheet each query point is on."""
    closest, which = closest_point_on_triangles(query, tri)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offset = np.einsum("ij,ij->i", query - closest, normals[which])
    side = np.sign(offset)
    side[np.abs(offset) < tol] = 0.0
    return side


def build_opening(points, hexes_old, faces, owners, cut_mask, inc, side):
    """Displacement that pulls the two sides of the crack apart."""
    centers = points[hexes_old].mean(axis=1)
    cut_faces, cut_owners = faces[cut_mask], owners[cut_mask]
    direction = centers[cut_owners[:, 1]] - centers[cut_owners[:, 0]]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # orient every cut face's normal from the minus side to the plus side
    direction *= side[cut_owners[:, 1]][:, None]

    accum = np.zeros((len(points) + inc["n_duplicated"], 3))
    weight = np.zeros(len(accum))
    n_cells = len(hexes_old)
    for owner_slot in (0, 1):
        cells = np.repeat(cut_owners[:, owner_slot], 4)
        nodes = cut_faces.ravel()
        copy = inc["new_node"][
            np.searchsorted(inc["keys"], nodes.astype(np.int64) * n_cells + cells)
        ]
        contribution = np.repeat(direction, 4, axis=0) * side[cells][:, None]
        np.add.at(accum, copy, contribution)
        np.add.at(weight, copy, 1.0)

    moving = weight > 0
    accum[moving] /= weight[moving][:, None]
    scale = np.median(
        np.linalg.norm(points[cut_faces[:, 1]] - points[cut_faces[:, 0]], axis=1)
    )
    return accum * OPENING * scale


def rim_nodes(cut_faces):
    """Nodes on the rim of the staircase surface -- its tip line.

    An edge used by a single cut face bounds the sheet; the crack closes
    along it, so the cells there legitimately keep sharing those nodes."""
    edges = np.sort(
        np.concatenate([cut_faces[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique_edges[counts == 1])


def color_sides(new_hexes, touched, rim):
    """Colour the cells around the crack by what they can still reach.

    Two touched cells are joined when they share a node that is NOT on the
    rim -- i.e. everywhere except the tip, where the crack is closed anyway.
    If the detachment worked, the colours come out as two groups that never
    mix across the surface."""
    node_lists = new_hexes[touched]
    keep = ~np.isin(node_lists, rim)
    rows, cols = np.nonzero(keep)
    nodes = node_lists[rows, cols]
    _, node_index = np.unique(nodes, return_inverse=True)
    node_index = node_index.ravel()

    n_cells, n_nodes = len(touched), node_index.max() + 1
    graph = coo_matrix(
        (np.ones(len(rows)), (rows, n_cells + node_index)),
        shape=(n_cells + n_nodes, n_cells + n_nodes),
    )
    n_comp, labels = connected_components(graph, directed=False)
    cell_labels = labels[:n_cells]
    _, colors = np.unique(cell_labels, return_inverse=True)
    return colors.ravel() + 1


def main():
    print(f"reading grid {GRID_PATH}", flush=True)
    grid = meshio.read(str(GRID_PATH))
    points = grid.points.astype(float)
    hexes = np.vstack(
        [block.data for block in grid.cells if block.type == "hexahedron"]
    )
    faces, owners = build_face_table(hexes)
    print(f"  {len(hexes):,} hexes, {len(points):,} points", flush=True)

    cut_files = sorted(CUT_DIR.glob("*_cut_faces.npz"))
    if not cut_files:
        raise SystemExit(f"no cut-face sets in {CUT_DIR} -- run cut_surface first")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cut_file in cut_files:
        name = cut_file.name.replace("_cut_faces.npz", "")
        data = np.load(cut_file)
        cut_mask = find_cut_faces(faces, data["face_nodes"])
        cut_owners = owners[cut_mask]
        print(f"\n{name}: {int(cut_mask.sum()):,} cut faces", flush=True)

        new_points, new_hexes, inc = split_nodes(points, hexes, faces, owners, cut_mask)
        print(
            f"  {inc['n_crack_nodes']:,} crack nodes -> {inc['n_duplicated']:,} "
            f"duplicated ({len(new_points):,} points now)",
            flush=True,
        )

        # --- verification -------------------------------------------------
        rim = rim_nodes(faces[cut_mask])
        rng = np.random.default_rng(0)
        checked = cut_owners
        if len(checked) > 20_000:
            checked = checked[rng.choice(len(checked), 20_000, replace=False)]
        shared = [np.intersect1d(new_hexes[a], new_hexes[b]) for a, b in checked]
        off_rim = np.array([int((~np.isin(s_, rim)).sum()) for s_ in shared])
        on_rim = np.array([len(s_) for s_ in shared]) - off_rim
        print(
            f"  cut pairs fully detached: {int((on_rim + off_rim == 0).sum())} / "
            f"{len(shared)} checked"
        )
        print(
            f"  cut pairs still joined along the tip rim (expected): "
            f"{int((on_rim > 0).sum())}"
        )
        print(f"  cut pairs joined AWAY from the rim (must be 0): {int((off_rim > 0).sum())}")

        weld = (owners[:, 1] >= 0) & ~cut_mask
        weld_owners = owners[weld]
        sample = rng.choice(
            len(weld_owners), size=min(2000, len(weld_owners)), replace=False
        )
        weld_shared = np.array(
            [
                len(np.intersect1d(new_hexes[a], new_hexes[b]))
                for a, b in weld_owners[sample]
            ]
        )
        print(
            f"  intact neighbours still sharing 4 nodes: "
            f"{int((weld_shared == 4).sum())} / {len(weld_shared)} sampled"
        )
        drift = np.linalg.norm(
            new_points[len(points) :] - points[inc["origin_of_copy"]], axis=1
        )
        print(
            "  copies coincident with their original: "
            f"{int((drift == 0).sum())} / {len(drift)}"
        )

        stl_path = SURFACE_DIR / f"{name}.stl"
        surface = meshio.read(str(stl_path))
        tri = surface.points[surface.cells_dict["triangle"]].astype(float)
        centers = points[hexes].mean(axis=1)
        touched = np.unique(cut_owners)
        side = np.zeros(len(hexes))
        side[touched] = geometric_side(centers[touched], tri)
        agree = side[cut_owners[:, 0]] * side[cut_owners[:, 1]] < 0
        print(
            f"  cut faces with the two cells on opposite sides: "
            f"{int(agree.sum())} / {len(agree)}"
        )

        # --- coloring: what can still reach what ----------------------------
        colors = color_sides(new_hexes, touched, rim)
        color_field = np.zeros(len(hexes))
        color_field[touched] = colors
        mixed = [
            c
            for c in np.unique(colors)
            if len(np.unique(side[touched][colors == c])) > 1
        ]
        print(
            f"  colours around the crack: {len(np.unique(colors))} "
            f"({', '.join(f'{int((colors == c).sum())} cells' for c in np.unique(colors))})"
        )
        print(f"  colours mixing the two geometric sides (must be 0): {len(mixed)}")

        # --- coloring + export --------------------------------------------
        opening = build_opening(
            points, hexes, faces, owners, cut_mask, inc, side
        )
        crack_cell = np.zeros(len(hexes))
        crack_cell[touched] = 1.0

        out_path = OUTPUT_DIR / f"{name}_detached.vtu"
        meshio.write_points_cells(
            out_path,
            new_points,
            [("hexahedron", new_hexes)],
            cell_data={
                "crack_side": [side],
                "crack_cell": [crack_cell],
                "side_color": [color_field],
            },
            point_data={"opening": opening},
        )
        print(f"  wrote {out_path}", flush=True)
        print(
            "  in ParaView: Warp By Vector on 'opening' -- the crack opens, "
            "the tip stays stitched",
            flush=True,
        )


if __name__ == "__main__":
    main()
