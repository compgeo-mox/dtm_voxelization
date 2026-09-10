"""Conformity / validity checks for the 2D quad grid (grid2d.py) and
the 3D hex grid it gets extruded into (extrude3d.py).

Both checks follow the same shape: every element has the right number
of distinct vertices (catches degenerate/collapsed elements), every
INTERNAL edge/face is shared by exactly 2 elements, every BOUNDARY
edge/face is shared by exactly 1 element AND lies on the domain's true
outer boundary (a boundary edge/face that ISN'T on the true boundary is
the direct signature of a hanging node -- this catches defects that
element-count or volume/area checks alone miss), and total area/volume
matches the domain exactly.
"""

from collections import defaultdict
import time

import numpy as np


def check_conformity(loops, xmin, xmax, ymin, ymax, tol=1e-9):
    """2D check for a list of quad loops (grid2d.py's output): every
    element is an actual quad (4 distinct vertices -- catches triangles
    or degenerate/collapsed shapes slipping in), every quad wound
    CONSISTENTLY (same signed-area sign -- catches a clockwise-wound
    loop hiding among the rest, invisible to every other check here
    since they all use absolute area/values, but not invisible to
    extrude3d.py: extruding a clockwise 2D quad produces a negative-
    volume hex; confirmed to happen in practice, from a single
    clockwise-wound branch in grid2d.small_square_transition, caught
    only once real DTM data reached geometry_checks.check_conformity_3d
    -- this check exists so that class of bug is caught here instead,
    two stages earlier and far cheaper to diagnose), every internal
    edge shared by exactly 2 quads, every boundary edge by exactly 1
    and lying on the true outer boundary, and total area matching the
    domain exactly.
    """

    def key(p):
        return (round(p[0], 9), round(p[1], 9))

    non_quads = [loop for loop in loops if len({key(p) for p in loop}) != 4]
    if non_quads:
        print(
            f"NON-QUAD elements: {len(non_quads)} (expected 4 distinct vertices each)"
        )
        for loop in non_quads[:5]:
            print("  ", loop)

    edge_count = defaultdict(int)
    total_area = 0.0
    signed_areas = []
    for loop in loops:
        n = len(loop)
        # shoelace area
        area = 0.0
        for i in range(n):
            x1, y1 = loop[i]
            x2, y2 = loop[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        signed_areas.append(area / 2)
        total_area += abs(area) / 2
        for i in range(n):
            a, b = key(loop[i]), key(loop[(i + 1) % n])
            edge_count[frozenset((a, b))] += 1

    n_negative = sum(1 for a in signed_areas if a < 0)
    n_positive = sum(1 for a in signed_areas if a > 0)
    inconsistent_winding = n_negative > 0 and n_positive > 0
    if inconsistent_winding:
        print(
            f"INCONSISTENT WINDING: {n_positive} CCW, {n_negative} CW quads "
            f"(should be all one or the other)"
        )

    mult = np.bincount(list(edge_count.values()))
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    phantom = 0
    for e in boundary_edges:
        pts = list(e)
        on_outer = all(
            abs(p[0] - xmin) < tol
            or abs(p[0] - xmax) < tol
            or abs(p[1] - ymin) < tol
            or abs(p[1] - ymax) < tol
            for p in pts
        )
        if not on_outer:
            phantom += 1

    expected_area = (xmax - xmin) * (ymax - ymin)
    print(f"quads: {len(loops)}")
    print(f"edge multiplicity dist: {mult}")
    print(
        f"total area: {total_area:.6f}  expected: {expected_area:.6f}  "
        f"diff: {total_area - expected_area:.2e}"
    )
    print(f"phantom (hanging-node) boundary edges: {phantom} / {len(boundary_edges)}")
    print(f"non-quad elements: {len(non_quads)}")
    print(f"winding: {n_positive} CCW / {n_negative} CW")
    return (
        not non_quads
        and not inconsistent_winding
        and phantom == 0
        and abs(total_area - expected_area) < 1e-6
    )


# ---------------------------------------------------------------------------
# 3D: hex volumes (shared with drape_mesh.py's Laplacian smoothing, which
# needs the same per-hex signed volume for its inversion-safety check) and
# the 3D analogue of check_conformity above.
# ---------------------------------------------------------------------------

# 6 tets covering a VTK_HEXAHEDRON (local corner order 0-3 bottom CCW, 4-7
# top CCW directly above 0-3), fanned from corner 0 and the far corner 6.
_HEX_TET_IDX = [
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
]

# The 6 quad faces of that same local corner order: bottom, top, then the
# 4 sides walking around (0,1,5,4) -> (1,2,6,5) -> (2,3,7,6) -> (3,0,4,7).
_HEX_FACES = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


def hex_volumes_signed(points, hexes):
    """Signed volume of every hex at once (vectorized -- no per-hex
    Python loop, since this can run many times per call, e.g. inside
    drape_mesh.py's laplacian_smooth per-iteration inversion check, and
    a loop would be far too slow for a mesh this size)."""
    points = np.asarray(points)
    hexes = np.asarray(hexes)
    coords = points[hexes]  # (n_hex, 8, 3)

    def tet_vol(a, b, c, d):
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0

    vol = np.zeros(len(hexes))
    for i, j, k, m in _HEX_TET_IDX:
        vol += tet_vol(coords[:, i], coords[:, j], coords[:, k], coords[:, m])
    return vol


def check_conformity_3d(points, hexes, xmin, xmax, ymin, ymax, zmin, zmax, tol=1e-6):
    """3D analogue of check_conformity, for the hex mesh extrude3d.py
    produces: every hex has 8 distinct vertices and strictly positive
    volume, every internal face shared by exactly 2 hexes, every
    boundary face by exactly 1 and lying on the domain's true outer
    box (a phantom boundary face is the 3D signature of a hanging
    node), and total volume matching the domain exactly.

    Faces are identified by the raw POINT INDICES at their 4 corners,
    not rounded coordinates like the 2D check_conformity uses -- valid
    as long as coincident points already share the same index, which
    holds for any mesh built by extrude3d.extrude_uniform (it
    deduplicates points by exact coordinate as it builds) or read back
    from a file written from one. This is what makes the check
    tractable at real mesh sizes (can be millions of hexes), where
    coordinate-rounding and dict-keying every face by its rounded
    coordinates would be far slower.

    `tol` is used both for the boundary-box membership test (absolute,
    in the same units as the coordinates) and, scaled by the domain's
    own expected volume, for the volume-conservation check -- a fixed
    tiny absolute volume tolerance is too strict for a real DTM-scale
    domain, where floating-point accumulation error over millions of
    hexes can easily exceed 1e-6 in absolute terms while still being
    utterly negligible relative to the domain's actual volume.

    Face identity/counting is done with a fully vectorized numpy
    sort+np.unique (an earlier version used a Python dict keyed by
    frozenset(point indices) per face -- correct, but its millions of
    Python frozenset/dict-entry objects at real mesh sizes (multi-
    million hexes) used tens of GB of RAM and got OOM-killed; plain
    int arrays instead keep this to a small multiple of the raw face
    data itself).
    """
    points = np.asarray(points)
    hexes = np.asarray(hexes)
    t0 = time.perf_counter()
    print(
        f"[conformity] start: {len(hexes):,} hexes, {len(points):,} points; "
        f"building {len(hexes) * 6:,} face records",
        flush=True,
    )

    sorted_hexes = np.sort(hexes, axis=1)
    degenerate_mask = (np.diff(sorted_hexes, axis=1) == 0).any(axis=1)
    n_degenerate = int(degenerate_mask.sum())
    if n_degenerate:
        print(f"DEGENERATE hexes: {n_degenerate} (expected 8 distinct vertices each)")

    vols = hex_volumes_signed(points, hexes)
    bad_vol = int((vols <= 0).sum())
    print(
        f"[conformity] volumes checked in {time.perf_counter() - t0:.1f}s; "
        "sorting and deduplicating faces",
        flush=True,
    )

    face_idx = np.asarray(_HEX_FACES)  # (6, 4)
    faces = hexes[:, face_idx].reshape(-1, 4)  # (n_hex * 6, 4)
    faces.sort(axis=1)
    uniq_faces, counts = np.unique(faces, axis=0, return_counts=True)
    print(
        f"[conformity] face uniqueness finished in {time.perf_counter() - t0:.1f}s; "
        f"{len(uniq_faces):,} unique faces",
        flush=True,
    )

    mult = np.bincount(counts)
    boundary_faces = uniq_faces[counts == 1]

    on_boundary_box = (
        (np.abs(points[:, 0] - xmin) < tol)
        | (np.abs(points[:, 0] - xmax) < tol)
        | (np.abs(points[:, 1] - ymin) < tol)
        | (np.abs(points[:, 1] - ymax) < tol)
        | (np.abs(points[:, 2] - zmin) < tol)
        | (np.abs(points[:, 2] - zmax) < tol)
    )
    phantom = int((~on_boundary_box[boundary_faces]).any(axis=1).sum())
    print(
        f"[conformity] boundary-face check finished in {time.perf_counter() - t0:.1f}s; "
        f"{len(boundary_faces):,} boundary faces",
        flush=True,
    )

    expected_volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    total_volume = float(vols.sum())
    volume_tol = max(tol, 1e-9 * abs(expected_volume))

    print(f"hexes: {len(hexes)}")
    print(f"face multiplicity dist: {mult}")
    print(f"negative/zero volume hexes: {bad_vol}")
    print(
        f"total volume: {total_volume:.4f}  expected: {expected_volume:.4f}  "
        f"diff: {total_volume - expected_volume:.4e}"
    )
    print(f"phantom (hanging-node) boundary faces: {phantom} / {len(boundary_faces)}")
    print(f"degenerate hexes: {n_degenerate}")

    return (
        n_degenerate == 0
        and bad_vol == 0
        and phantom == 0
        and abs(total_volume - expected_volume) < volume_tol
    )
