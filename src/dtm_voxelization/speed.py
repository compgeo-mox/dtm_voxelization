"""Export the detached grid to SPEED's mesh format, with labelled boundary quads.

SPEED reads the ASCII layout CubitPython4SPEED's `utilities.export_mesh`
writes (the `GRIDFILE` of a SPEED case):

    num_nodes  num_elements  0  0  0
    node_id  x  y  z                          (1-based)
    quad_id  tag  quad  n1 n2 n3 n4           (boundary faces first)
    hex_id   tag  hex   n1 ... n8             (then the cells)

Hexes go in VTK/Exodus order with a positive Jacobian at every corner, quads
are wound with their normal pointing out of the mesh -- both checked here
before writing. Quad tags are the labels the `.mate` file assigns conditions
to (e.g. `ABSO 2`).

Boundary faces are the faces with a single owner; wound as that owner's local
face they already point outward. They are labelled by where they sit:

- LATERAL_BOTTOM_TAG: on the bounding-box planes x = xmin/xmax, y = ymin/ymax
  and z = zmin;
- TOP_TAG: every other boundary face -- the carved terrain and both sides of
  every crack, which carry the same condition.

Crack sides are still identified, as boundary faces with a geometrically
coincident twin (only the detachment creates those), to check their number
against the cut faces of all the case's surfaces.

Writes `speed/<case>.mesh`, plus `speed/<case>_boundary.vtu` with the quads
and their tags for a look in ParaView.
"""

import numpy as np
import meshio

from .mesh import build_face_table, read_hex_mesh, require

HEX_TAG = 1
LATERAL_BOTTOM_TAG = 2
TOP_TAG = 3

# corner -> its three neighbours, ordered as a right-handed frame in VTK order
CORNER_FRAMES = np.array(
    [[1, 3, 4], [2, 0, 5], [3, 1, 6], [0, 2, 7], [7, 5, 0], [4, 6, 1], [5, 7, 2], [6, 4, 3]]
)


def count_inverted_hexes(points, hexes, chunk=500_000):
    """Hexes with a non-positive Jacobian at any of their 8 corners."""
    bad = 0
    for lo in range(0, len(hexes), chunk):
        x = points[hexes[lo : lo + chunk]]
        det = np.linalg.det(x[:, CORNER_FRAMES, :] - x[:, :, None, :])
        bad += int((det.min(axis=1) <= 0).sum())
    return bad


def coincident_twins(quad_coords):
    """Boundary faces lying exactly on another boundary face.

    The detachment copies coordinates bit for bit, so the two sides of a crack
    share the same four points in a different node order: sorting the points
    of each face gives an exact key."""
    order = np.lexsort(
        (quad_coords[..., 2], quad_coords[..., 1], quad_coords[..., 0]), axis=-1
    )
    key = np.take_along_axis(quad_coords, order[..., None], axis=1)
    _, inverse, counts = np.unique(
        key.reshape(len(quad_coords), 12), axis=0, return_inverse=True, return_counts=True
    )
    counts = counts[inverse.ravel()]
    if counts.max() > 2:
        raise ValueError(f"{int((counts > 2).sum())} boundary faces coincide with more than one other")
    return counts == 2


def count_cut_faces(case):
    """Distinct cut faces over all the case's surfaces (a face cut by two
    surfaces is split once)."""
    keys = []
    for surface in case.surfaces:
        path = case.cut_dir / f"{surface.name}.npz"
        require(path, "cut")
        keys.append(np.sort(np.load(path)["face_nodes"], axis=1))
    if sum(len(k) for k in keys) == 0:
        return 0
    return len(np.unique(np.vstack(keys), axis=0))


def write_speed_mesh(path, points, quads, quad_tags, hexes):
    """SPEED ASCII mesh: header, nodes, quads, hexes, all 1-based."""
    with open(path, "w") as f:
        f.write(f"  {len(points)}  {len(quads) + len(hexes)}  0  0  0\n")
        np.savetxt(
            f,
            np.column_stack([np.arange(1, len(points) + 1), points]),
            fmt="%d  %+.7e  %+.7e  %+.7e",
        )
        np.savetxt(
            f,
            np.column_stack([np.arange(1, len(quads) + 1), quad_tags, quads + 1]),
            fmt="%d  %d  quad  " + "  ".join(["%d"] * 4),
        )
        np.savetxt(
            f,
            np.column_stack(
                [np.arange(1, len(hexes) + 1), np.full(len(hexes), HEX_TAG), hexes + 1]
            ),
            fmt="%d  %d  hex  " + "  ".join(["%d"] * 8),
        )


def run(case):
    """detached.vtu -> speed/<case>.mesh with labelled boundary quads."""
    require(case.detached_path, "detach")
    points, hexes = read_hex_mesh(case.detached_path)
    print(
        f"  detached grid {case.detached_path}: {len(hexes):,} hexes, {len(points):,} points",
        flush=True,
    )

    inverted = count_inverted_hexes(points, hexes)
    print(f"  hexes with a non-positive corner Jacobian: {inverted}", flush=True)
    if inverted:
        raise ValueError(f"{inverted} inverted hexes -- SPEED needs positive Jacobians")

    faces, owners = build_face_table(hexes)
    boundary = owners[:, 1] < 0
    quads, quad_owner = faces[boundary], owners[boundary, 0]
    print(f"  {len(faces):,} faces, {len(quads):,} on the boundary", flush=True)

    q = points[quads]
    face_centers = q.mean(axis=1)
    normals = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    outward = np.einsum(
        "ij,ij->i", normals, face_centers - points[hexes[quad_owner]].mean(axis=1)
    )
    inward = int((outward <= 0).sum())
    print(f"  boundary quads wound inward: {inward}", flush=True)
    if inward:
        raise ValueError(f"{inward} boundary quads point into the mesh")

    lo, hi = points.min(axis=0), points.max(axis=0)
    tol = 1e-6 * (hi - lo).max()
    print(f"  bbox {lo.round(3)} .. {hi.round(3)}, plane tolerance {tol:.2e}", flush=True)
    lateral = (
        (np.abs(face_centers[:, 0] - lo[0]) < tol)
        | (np.abs(face_centers[:, 0] - hi[0]) < tol)
        | (np.abs(face_centers[:, 1] - lo[1]) < tol)
        | (np.abs(face_centers[:, 1] - hi[1]) < tol)
    )
    bottom = np.abs(face_centers[:, 2] - lo[2]) < tol
    crack = coincident_twins(q)
    print(
        f"  lateral {int(lateral.sum()):,}, bottom {int(bottom.sum()):,}, "
        f"crack {int(crack.sum()):,} ({int(crack.sum()) // 2:,} twin pairs), "
        f"crack faces on the bbox planes {int((crack & (lateral | bottom)).sum())}",
        flush=True,
    )

    n_cut = count_cut_faces(case)
    print(f"  cut faces over all surfaces: {n_cut:,}", flush=True)
    if n_cut != int(crack.sum()) // 2:
        raise ValueError(
            f"{int(crack.sum()) // 2} twin pairs but {n_cut} cut faces -- "
            "detached.vtu and cut_faces/ come from different runs"
        )

    tags = np.full(len(quads), TOP_TAG)
    tags[(lateral | bottom) & ~crack] = LATERAL_BOTTOM_TAG
    order = np.argsort(tags, kind="stable")
    quads, tags = quads[order], tags[order]
    for tag, label in [(LATERAL_BOTTOM_TAG, "lateral+bottom"), (TOP_TAG, "top+cracks")]:
        print(f"  tag {tag} ({label}): {int((tags == tag).sum()):,} quads", flush=True)

    case.speed_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = case.speed_dir / f"{case.name}.mesh"
    write_speed_mesh(mesh_path, points, quads, tags, hexes)
    print(
        f"  wrote {mesh_path} -- {len(points):,} nodes, "
        f"{len(quads):,} quads + {len(hexes):,} hexes",
        flush=True,
    )

    used, compact = np.unique(quads, return_inverse=True)
    vtu_path = case.speed_dir / f"{case.name}_boundary.vtu"
    meshio.write_points_cells(
        vtu_path,
        points[used],
        [("quad", compact.reshape(-1, 4))],
        cell_data={"tag": [tags.astype(float)]},
    )
    print(f"  wrote {vtu_path}", flush=True)
