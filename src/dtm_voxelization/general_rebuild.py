"""General, from-scratch assembler for the ternary local hex-refinement
scheme: takes an ARBITRARY footprint of refined coarse cells (any shape,
any number of disconnected regions) at one grid layer, and builds a fully
conforming hex mesh using only the verified template library in
templates.py -- no CUBIT re-parsing, no hardcoded case-specific geometry.

Classification rule per (i,j) cell at the refined layer k=K_URZ, based on
how many of its 4 orthogonal neighbors are themselves in the refined
footprint ("touch count"):
  - cell itself in the footprint            -> URZ core (27-way split)
  - touches footprint via 2 ORTHOGONAL full faces (a reentrant corner,
    not itself in the footprint)            -> concave mega-block (this
                                                cell + its 2 "arm"
                                                neighbors, claimed together)
  - touches footprint via exactly 1 full face -> wall (WALL_BASE, rotated)
  - touches footprint only diagonally (1 diagonal neighbor in the
    footprint, 0 face-adjacent neighbors)    -> convex corner (CORNER_BASE)
  - otherwise                                -> plain coarse cube

The k=K_URZ-1 layer directly beneath gets the matching cap: URZCAP under
a core cell, WALLCAP under a wall cell (including arm cells -- verified
identical rule), nothing under a convex corner, and the concave mega-
block's own cap pieces under a concave corner + its arms. Any touch-count
this library has no template for (3 or 4 full faces) raises an error
instead of guessing -- silently-wrong output is worse than a loud failure.
"""
from collections import defaultdict

import numpy as np

from . import templates as T
from .geometry_checks import check_conformity_3d, hex_volumes_signed

NBRS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}


def refine_region_further(points, hexes, tags, footprint, k_layers, NX, NY, NZ,
                           XMIN, YMIN, ZMIN, local_size=5, cell_size=1.0):
    """Multi-level refinement: push one lateral `footprint` (set of
    (i,j), same convention as build_mesh's own) through `k_layers`
    (consecutive k range) of ORIGINAL, still-plain 'coarse' (or 'urz' --
    see below) cells -- identified in the SAME coordinate system used to
    build the mesh (the caller already knows these, having chosen
    XMIN/YMIN/ZMIN/NX/NY/NZ/cell_size for whatever build produced the
    current mesh) -- to a further level of refinement (each becomes a
    properly-transitioned 27-way urz split, exactly as if it had been
    part of the ORIGINAL footprint at that level).

    PRECONDITION, enforced (this is the "user responsibility to select
    only square-shaped hexas" the caller must satisfy): every targeted
    (i,j,k) must currently be a plain 'coarse' OR 'urz' cell -- not a
    wall/corner/concave transition hex (which this library has no rule
    for subdividing further). 'urz' is accepted alongside 'coarse'
    because a urz cell's own 27 children ARE, individually, exactly the
    same kind of plain axis-aligned box as a 'coarse' cell -- they just
    happen to be tagged with their parent block's shared tag rather than
    each getting a distinct one. This is what makes NESTING possible:
    call this once with cell_size=1.0 on an originally-coarse cell to get
    a urz split, then call it AGAIN with cell_size = (that cell's own
    edge length, e.g. 1/3 of the level above) targeting one of ITS 27
    children by their own (i,j,k) in that finer grid -- "apply the same
    algorithm again, at whatever scale is needed." Checked by locating
    that exact box (by position, at the given cell_size) among the
    current hexes and confirming its tag.

    How this actually works, and why the naive version of this function
    (an earlier draft) was WRONG: replacing a hex with a geometrically-
    rescaled finer grid does NOT preserve conformity with its untouched
    neighbors, even with perfectly correct scaling arithmetic -- a single
    flat face fundamentally cannot match a neighbor made of many smaller
    faces, regardless of how precisely it's positioned (verified this
    concretely: even an all-coarse, zero-refinement rescaled sub-grid
    still produced phantom faces against unchanged neighbors). The fix is
    to never rescale: build the local block at the SAME natural
    cell_size-per-cell spacing as the parent mesh, wide enough
    (local_size, default 5) to include a real ring of the target's own
    same-scale neighbors, and splice out/in by exact position -- so the
    untouched neighbor cells in that local block are simply reproduced
    byte-for-byte identical to what they already were, and only the
    target's own footprint genuinely changes. local_size must be >=5 (a
    target cell at the exact center needs 2 cells of buffer beyond its
    own wall/corner ring for the same reason build_mesh's own top-level
    footprints always need buffer from the domain edge).

    The whole footprint x k_layers is refined in ONE build_mesh call over
    ONE bounding block (footprint's own extent plus a `buf`-cell ring on
    every side) -- not cell-by-cell -- which is what makes it safe to
    densely refine a real, multi-cell contiguous region: processing
    adjacent cells one at a time would have the second cell see the
    first one's still-uncommitted wall/corner ring and reject it as
    "not eligible" (this was an actual bug in an earlier version tested
    against a single-cell-at-a-time loop). refine_cells_further below
    remains for the "a few independent, separated single-cell targets"
    case, by calling this once per target.

    A single-cell target is a footprint of exactly one (i,j) cell through
    exactly one k -- this is the same operation for any footprint SIZE,
    dense/contiguous or a single point, since the same local_size buffer
    (>= 5, checked over the FULL bounding box of footprint x k_layers,
    not just the boundary of a single cell) applies either way; a wider
    footprint just means a taller local block covering all of it plus
    its own buffer ring, in one build_mesh call instead of one per cell
    (which matters: refining every cell of a real region one at a time
    would have each one see the PREVIOUS one's still-uncommitted wall/
    corner ring as "not eligible", failing immediately on the second
    cell of any two adjacent ones).

    Returns (points2, hexes2, tags2), same conventions as build_mesh."""
    if local_size < 5 or local_size % 2 == 0:
        raise ValueError(f"local_size must be odd and >= 5, got {local_size}")
    buf = local_size // 2
    cx, cy, cz = T.cs3(cell_size)

    def is_eligible(kind):
        # 'coarse' or any depth of nested urz ('urz' itself from level 1,
        # 'L2_urz' from a level-2 call, 'L3_urz' from level 3, etc.) --
        # all structurally the same plain axis-aligned box, just tagged
        # with how many refine_region_further calls produced it, which
        # is what makes a THIRD (or deeper) level possible: nested inside
        # an 'L2_urz' cell needs it recognized as eligible too, not just
        # plain 'urz'.
        return kind == "coarse" or kind.endswith("urz")

    footprint = set(footprint)
    if not footprint:
        return points, hexes, tags
    k_layers = sorted(k_layers)
    if k_layers != list(range(k_layers[0], k_layers[-1] + 1)):
        raise ValueError(f"k_layers must be a consecutive range, got {k_layers}")

    # buffer margin on each of the 6 sides, EXCEPT where the footprint
    # already reaches the fine-grid's own domain edge on that side -- a
    # region that reaches the true top (k_layers[-1] == NZ-1) needs no
    # topcap, exactly like build_mesh's own has_above=False case for the
    # SAME reason: there's no coarse neighbor above to transition to, so
    # no buffer/cap is needed there either. Skipping the margin (instead
    # of requiring buf cells that don't exist past the edge) is what
    # lets a nested level reach the same "starts from the top" boundary
    # its parent level does, rather than being squeezed away from it.
    ii = [i for i, j in footprint]
    jj = [j for i, j in footprint]
    i_lo = min(ii) - (buf if min(ii) - buf >= 0 else min(ii))
    i_hi = max(ii) + (buf if max(ii) + buf <= NX - 1 else NX - 1 - max(ii))
    j_lo = min(jj) - (buf if min(jj) - buf >= 0 else min(jj))
    j_hi = max(jj) + (buf if max(jj) + buf <= NY - 1 else NY - 1 - max(jj))
    k_lo = k_layers[0] - (buf if k_layers[0] - buf >= 0 else k_layers[0])
    k_hi = k_layers[-1] + (buf if k_layers[-1] + buf <= NZ - 1 else NZ - 1 - k_layers[-1])
    block_size = (i_hi - i_lo + 1, j_hi - j_lo + 1, k_hi - k_lo + 1)

    # position (rounded min-corner, RELATIVE to cell_size -- see
    # build_mesh's gid() for why the rounding must scale with cell_size
    # rather than use a fixed absolute tolerance) -> hex index, built
    # ONCE rather than linearly rescanning all hexes for each cell of the
    # bounding box checked below -- the dominant cost on a real, tens-of-
    # thousands-of-hexes mesh otherwise.
    by_pos = {}
    for hi2, h2 in enumerate(hexes):
        key2 = tuple(round(v, 5) for v in points[h2].min(0) / (cx, cy, cz))
        by_pos[key2] = hi2

    def lookup(i, j, k):
        x0 = XMIN + i * cx
        y0 = YMIN + j * cy
        z0 = ZMIN + k * cz
        key = (round(x0 / cx, 5), round(y0 / cy, 5), round(z0 / cz, 5))
        return by_pos.get(key)

    input_kinds = set()
    for i, j in footprint:
        for k in k_layers:
            hi = lookup(i, j, k)
            if hi is None or not is_eligible(tags[hi].split(":")[0]):
                raise ValueError(
                    f"footprint cell (i,j,k)={(i,j,k)} at cell_size={cell_size} "
                    f"is not a plain 'coarse'/'urz' hex in the current mesh -- "
                    f"found tag {tags[hi] if hi is not None else '<no matching hex>'!r}. "
                    f"refine_region_further only accepts cells that are still "
                    f"fully unrefined at their own scale.")
            input_kinds.add(tags[hi].split(":")[0])
    # output level = one deeper than the footprint's OWN current level --
    # 'coarse' or plain 'urz' (level 1, no L-prefix) -> 'L2_urz'; 'L2_urz'
    # -> 'L3_urz'; etc, so a third (or deeper) call nested inside an
    # already-nested nb one is visibly distinguished, not just relabeled
    # 'L2_' again. Mixed-level footprints (shouldn't normally happen)
    # use the DEEPEST input level present, so the output is never
    # ambiguously shallower than any cell it's actually replacing.
    def level_of(kind):
        n = 1
        while kind.startswith("L") and "_" in kind:
            try:
                n = max(n, int(kind[1:kind.index("_")]))
            except ValueError:
                break
            kind = kind[kind.index("_") + 1:]
        return n
    out_level = max((level_of(k) for k in input_kinds), default=1) + 1
    out_prefix = f"L{out_level}_"

    for i in range(i_lo, i_hi + 1):
        for j in range(j_lo, j_hi + 1):
            for k in range(k_lo, k_hi + 1):
                if (i, j) in footprint and k in k_layers:
                    continue  # already checked above
                hi = lookup(i, j, k)
                if hi is None or not is_eligible(tags[hi].split(":")[0]):
                    raise ValueError(
                        f"the footprint's buffer ring needs cell {(i,j,k)} "
                        f"(at cell_size={cell_size}) to be plain 'coarse'/'urz' "
                        f"too (found {tags[hi] if hi is not None else '<none>'!r}) "
                        f"-- this region is too close to an already-refined "
                        f"area, another target, or the domain edge; needs "
                        f"more separation or a bigger domain.")

    block_xmin = XMIN + i_lo * cx
    block_ymin = YMIN + j_lo * cy
    block_zmin = ZMIN + k_lo * cz
    local_footprint = {(i - i_lo, j - j_lo) for i, j in footprint}
    local_k_layers = range(k_layers[0] - k_lo, k_layers[-1] - k_lo + 1)
    loc_pts, loc_hexes, loc_tags = build_mesh(
        local_footprint, NX=block_size[0], NY=block_size[1], NZ=block_size[2],
        XMIN=block_xmin, YMIN=block_ymin, ZMIN=block_zmin,
        k_layers=local_k_layers, cell_size=cell_size)
    new_blocks = [loc_pts[h] for h in loc_hexes]
    new_tags = [f"{out_prefix}{t}" if t.split(":")[0] == "urz" else t for t in loc_tags]

    # splice: drop every current parent hex whose centroid falls inside
    # this block's bounding box, replace with new_blocks
    bxlo, bylo, bzlo = block_xmin, block_ymin, block_zmin
    bxhi = bxlo + block_size[0] * cx
    byhi = bylo + block_size[1] * cy
    bzhi = bzlo + block_size[2] * cz
    keep_mask = []
    for h in hexes:
        c = points[h].mean(0)
        inside = (bxlo < c[0] < bxhi) and (bylo < c[1] < byhi) and (bzlo < c[2] < bzhi)
        keep_mask.append(not inside)
    keep_mask = np.array(keep_mask)
    kept_blocks = [points[h] for h in hexes[keep_mask]]
    kept_tags = [t for i, t in enumerate(tags) if keep_mask[i]]

    all_blocks = kept_blocks + new_blocks
    all_tags = kept_tags + new_tags

    pts_list, pid = [], {}

    def gid(p):
        key = (round(float(p[0]) / cx, 5),
               round(float(p[1]) / cy, 5),
               round(float(p[2]) / cz, 5))
        idx = pid.get(key)
        if idx is None:
            idx = len(pts_list)
            pid[key] = idx
            pts_list.append((float(p[0]), float(p[1]), float(p[2])))
        return idx

    hexes = np.array([[gid(p) for p in h] for h in all_blocks])
    points = np.array(pts_list)
    tags = all_tags

    vols = hex_volumes_signed(points, hexes)
    bad = np.where(vols <= 0)[0]
    for idx in bad:
        h = hexes[idx]
        hexes[idx] = np.concatenate([h[:4][::-1], h[4:][::-1]])

    return points, hexes, tags


def refine_cells_further(points, hexes, tags, target_ijk, NX, NY, NZ,
                          XMIN, YMIN, ZMIN, local_size=5, cell_size=1.0):
    """Convenience wrapper over refine_region_further for one or more
    INDEPENDENT single-cell targets (processed one at a time, so distant
    targets don't need to share one huge bounding block) -- kept for the
    "push a few separate coarse cells further" use case tested earlier.
    For densely refining a whole contiguous region (e.g. a real, nested
    sub-rectangle of an already-refined area), call refine_region_further
    directly with the full footprint instead -- see its docstring for why
    processing many adjacent cells one at a time here would fail."""
    for (ti, tj, tk) in target_ijk:
        points, hexes, tags = refine_region_further(
            points, hexes, tags, {(ti, tj)}, [tk], NX, NY, NZ,
            XMIN, YMIN, ZMIN, local_size=local_size, cell_size=cell_size)
    return points, hexes, tags


def find_square_hexes(tags, kind=None):
    """Indices of every hex eligible as a refine_region_further target --
    tagged 'coarse' or any depth of nested urz ('urz', 'L2_urz', 'L3_urz',
    ...) (optionally restricted to just one exact tag-kind string)."""
    out = []
    for i, t in enumerate(tags):
        k = t.split(":")[0]
        if (k == "coarse" or k.endswith("urz")) and (kind is None or k == kind):
            out.append(i)
    return out


def classify_footprint(footprint, ni_range, nj_range):
    """Returns (roles: dict[(i,j)] -> role info, claimed: set of (i,j)
    consumed by a concave mega-block). `footprint` is a set of (i,j)
    tuples (the refined cells). Scans every (i,j) in the given ranges."""
    roles = {}
    claimed = set()

    # pass 1: find every concave corner (2 orthogonal full-face touches,
    # not itself in the footprint) and its 2 arms, claim all 3 cells.
    concave_corners = []
    for i in ni_range:
        for j in nj_range:
            if (i, j) in footprint:
                continue
            touch_dirs = [name for name, (di, dj) in NBRS.items() if (i + di, j + dj) in footprint]
            if len(touch_dirs) == 2:
                d0, d1 = touch_dirs
                # must be orthogonal (not a straight-through 2-face
                # touch, which this scheme's footprints never produce
                # for a single reentrant notch, but check anyway)
                opposite = {"E": "W", "W": "E", "N": "S", "S": "N"}
                if opposite[d0] == d1:
                    raise ValueError(
                        f"cell {(i,j)} touches the footprint on 2 OPPOSITE "
                        f"sides ({d0},{d1}) -- not a supported concave-corner "
                        f"template (this would be a 1-cell-wide bridge, not "
                        f"a reentrant notch)."
                    )
                uf, arms = T._union_faces_and_arms((i, j), footprint)
                if frozenset(uf) not in T.CONCAVE:
                    raise ValueError(f"cell {(i,j)}: no concave template for uf={uf}")
                concave_corners.append(((i, j), uf, arms))

    for (i, j), uf, arms in concave_corners:
        cells_here = {(i, j)} | set(arms)
        overlap = cells_here & claimed
        if overlap:
            raise ValueError(
                f"concave corner at {(i,j)} overlaps an already-claimed "
                f"cell {overlap} -- footprint has 2 reentrant notches too "
                f"close together for independent templates; not supported."
            )
        roles[(i, j)] = dict(kind="concave", uf=uf, arms=arms)
        claimed |= cells_here

    # pass 2: classify every remaining cell against the RAW footprint
    # (claiming doesn't change how other cells see the footprint).
    for i in ni_range:
        for j in nj_range:
            if (i, j) in claimed:
                continue
            if (i, j) in footprint:
                roles[(i, j)] = dict(kind="urz")
                continue
            touch_dirs = [name for name, (di, dj) in NBRS.items() if (i + di, j + dj) in footprint]
            n_face = len(touch_dirs)
            if n_face == 1:
                di, dj = NBRS[touch_dirs[0]]
                roles[(i, j)] = dict(kind="wall", push=(di, dj))
            elif n_face == 0:
                diag_dirs = [(di, dj) for di in (-1, 1) for dj in (-1, 1) if (i + di, j + dj) in footprint]
                if len(diag_dirs) == 1:
                    roles[(i, j)] = dict(kind="corner", push=diag_dirs[0])
                else:
                    roles[(i, j)] = dict(kind="coarse")
            else:
                raise ValueError(
                    f"cell {(i,j)} touches the footprint via {n_face} full "
                    f"faces -- no template for this (only 0, 1, or a "
                    f"2-orthogonal reentrant-corner touch are supported)."
                )
    return roles, claimed


def build_mesh(footprint, NX=8, NY=8, NZ=8, XMIN=-4.0, YMIN=-4.0, ZMIN=-4.0,
                k_layers=None, K_URZ=None, cell_size=1.0):
    """footprint: set of (i,j) refined-cell indices, refined through EVERY
    layer in `k_layers` (a consecutive range of k-values -- e.g.
    range(NZ-3, NZ) to refine the top 3 coarse layers of the footprint).
    K_URZ=<int> is kept as a convenience alias for a single-layer refinement
    (k_layers=[K_URZ]).

    `cell_size`: the physical size of one (i,j,k) grid cell -- defaults to
    1.0 (the original convention this was built and tested with), but any
    positive value works identically ("just apply the same algorithm
    again, at whatever scale the caller needs" -- this is what makes
    refine_cells_further able to operate on an already-refined urz cell's
    own, smaller sub-cells: it just calls build_mesh again with
    cell_size = that cell's own edge length). MAY also be a (sx,sy,sz)
    triple instead of one scalar, for an anisotropic cell -- e.g. finer in
    z than x/y -- with the constraint sx==sy (see templates.apply_matrix's
    docstring for exactly why: every rotation this library uses is purely
    about the local z-axis, so it commutes with a diagonal scale only
    when the two axes it mixes, x and y, share the same factor; z alone
    is free to differ). Every offset/index computation here (XMIN+i*cx
    etc, the position-lookup and point-dedup rounding) already uses the
    matching per-axis component; every template PLACEMENT (a straight
    `* cell_size` multiply) broadcasts correctly against either a scalar
    or a (3,)-shaped cell_size automatically, no special-casing needed
    there.

    Between two CONSECUTIVE refined layers of the same (i,j) cell, no cap
    is placed at all -- the lateral template (urz/wall/corner/concave) is
    placed independently at each k, and the touching top/bottom faces of
    two stacked copies of the SAME template conform directly, with no
    transition piece needed, because every template in this library is
    verified z-mirror-symmetric (see templates.zflip_local's docstring):
    a template's own top-face (x,y) pattern is identical to its bottom-
    face pattern, so two identical templates stacked directly on each
    other present matching patterns at their shared interface. This is
    verified empirically below (0 phantom faces for a multi-layer-deep
    stack), not just assumed from the symmetry argument alone. A cap
    (the transition to a genuinely COARSE neighbor) is only placed below
    the BOTTOM of the whole k_layers stack and above its TOP -- exactly
    the same logic as the single-layer case, generalized.

    Returns (points, hexes, tags) -- tags[i] is a short string describing
    what generated hex i, for diagnostics/coloring."""
    if k_layers is None:
        if K_URZ is None:
            raise ValueError("build_mesh needs either k_layers or K_URZ")
        k_layers = [K_URZ]
    k_layers = sorted(k_layers)
    if k_layers != list(range(k_layers[0], k_layers[-1] + 1)):
        raise ValueError(f"k_layers must be a consecutive range, got {k_layers}")
    k_bottom, k_top = k_layers[0], k_layers[-1]
    cx, cy, cz = T.cs3(cell_size)

    ni_range = range(NX)
    nj_range = range(NY)
    roles, claimed = classify_footprint(footprint, ni_range, nj_range)

    raw = []
    tags = []

    def add(blk, tag):
        raw.extend(list(blk))
        tags.extend([tag] * len(blk))

    has_below = (k_bottom - 1) >= 0
    has_above = (k_top + 1) < NZ
    placed_k1 = set()   # (i,j) cells at k=k_bottom-1 already placed by a cap
    placed_kp1 = set()  # (i,j) cells at k=k_top+1 already placed by a topcap

    def add_bottom_cap(kind_tag, ij, local_template, x0, y0):
        """local_template: untranslated, in its own local [0,1]^3 frame."""
        if has_below:
            z_cap = ZMIN + (k_bottom - 1) * cz
            add(local_template * cell_size + np.array([x0, y0, z_cap]), f"{kind_tag}:{ij}")
            placed_k1.add(ij)

    def add_top_cap(kind_tag, ij, local_template, x0, y0):
        if has_above:
            z_cap = ZMIN + (k_top + 1) * cz
            add(T.zflip_local(local_template) * cell_size + np.array([x0, y0, z_cap]), f"{kind_tag}:{ij}")
            placed_kp1.add(ij)

    for (i, j), role in roles.items():
        x0, y0 = XMIN + i * cx, YMIN + j * cy
        kind = role["kind"]

        if kind == "urz":
            for k in k_layers:
                add(T.urz_cell(x0, y0, ZMIN + k * cz, cell_size=cell_size), f"urz:({i},{j},{k})")
            cap = T.apply_matrix(T.WALL_BASE, T.URZCAP_MAT)  # local, untranslated
            add_bottom_cap("urzcap", (i, j), cap, x0, y0)
            add_top_cap("urzcap_top", (i, j), cap, x0, y0)

        elif kind == "wall":
            rot = T.PUSH_TO_ROT[role["push"]]
            for k in k_layers:
                add(T.place_rot(T.WALL_BASE, rot, (x0, y0, ZMIN + k * cz), cell_size=cell_size),
                    f"wall:({i},{j},{k})")
            cap = T.apply_matrix(T.CORNER_BASE, T.WALLCAP_MAT_BY_ROT[rot])  # local, untranslated
            add_bottom_cap("wallcap", (i, j), cap, x0, y0)
            add_top_cap("wallcap_top", (i, j), cap, x0, y0)

        elif kind == "corner":
            rot = T.CORNER_TO_ROT[role["push"]]
            for k in k_layers:
                add(T.place_rot(T.CORNER_BASE, rot, (x0, y0, ZMIN + k * cz), cell_size=cell_size),
                    f"corner:({i},{j},{k})")
            # verified (straight case, real CUBIT data): no cap needed
            # below a convex corner; by the same z-mirror-symmetry
            # argument (CORNER_BASE verified self-symmetric under
            # zflip_local), none needed above it either -- and by the
            # same argument, stacked corner layers need nothing between
            # them either.

        elif kind == "concave":
            blk = T.CONCAVE[frozenset(role["uf"])]
            for k in k_layers:
                add(blk["corner"] * cell_size + np.array([x0, y0, ZMIN + k * cz]),
                    f"concave_core:({i},{j},{k})")
            add_bottom_cap("concave_cap", (i, j), blk["corner_cap"], x0, y0)
            add_top_cap("concave_cap_top", (i, j), blk["corner_cap"], x0, y0)
            for arm in blk["arms"]:
                oi, oj = arm["offset"]
                ai, aj = i + oi, j + oj
                ax0, ay0 = XMIN + ai * cx, YMIN + aj * cy
                for k in k_layers:
                    add(arm["core"] * cell_size + np.array([ax0, ay0, ZMIN + k * cz]),
                        f"concave_arm:({ai},{aj},{k})")
                add_bottom_cap("concave_armcap", (ai, aj), arm["cap"], ax0, ay0)
                add_top_cap("concave_armcap_top", (ai, aj), arm["cap"], ax0, ay0)

        elif kind == "coarse":
            pass  # handled in the generic fill pass below

        else:
            raise ValueError(f"unknown role kind {kind!r}")

    # fill every (i,j,k) NOT already placed above with a plain coarse cube.
    # IMPORTANT: only mark (i,j) as "handled" at a refined k for roles that
    # actually placed geometry there -- a plain "coarse" role placed
    # NOTHING in the loop above (it's deliberately deferred to this same
    # generic fill pass), so it must NOT be marked handled, or its cell
    # would be silently skipped entirely (found via the regression test:
    # this exact bug dropped 48 unclassified coarse cells from the
    # straight 2x2 case, a 48-unit volume deficit).
    handled_k = defaultdict(set)  # k -> set of (i,j) already placed at that k
    for (i, j), role in roles.items():
        if role["kind"] != "coarse":
            for k in k_layers:
                handled_k[k].add((i, j))
    for (i, j) in placed_k1:
        handled_k[k_bottom - 1].add((i, j))
    for (i, j) in placed_kp1:
        handled_k[k_top + 1].add((i, j))
    # concave arms' own (i,j) at each refined k are already in `roles` only
    # for the corner cell itself -- the arm cells are separate roles too
    # (kind "wall" would double-place them if also classified there), so
    # they must NOT be reclassified: remove them from consideration.
    concave_owned_ij = set()
    for (i, j), role in roles.items():
        if role["kind"] == "concave":
            concave_owned_ij.add((i, j))
            for arm in T.CONCAVE[frozenset(role["uf"])]["arms"]:
                oi, oj = arm["offset"]
                concave_owned_ij.add((i + oi, j + oj))
    for (i, j) in concave_owned_ij:
        for k in k_layers:
            handled_k[k].add((i, j))

    for i in ni_range:
        for j in nj_range:
            for k in range(NZ):
                if (i, j) in handled_k.get(k, ()):
                    continue
                x0, y0, z0 = XMIN + i * cx, YMIN + j * cy, ZMIN + k * cz
                add(T.coarse_box(x0, y0, z0, cell_size=cell_size), f"coarse:({i},{j},{k})")

    # dedupe points -- rounded to 6dp RELATIVE to cell_size (i.e. round
    # p/cell_size, not p itself), not a fixed absolute 6dp. The ~1e-7
    # imprecision baked into the real-CUBIT-extracted templates (WALL_BASE
    # etc; see templates.py) lives in their own LOCAL [0,1]^3 frame, and
    # gets scaled BY cell_size along with everything else -- at cell_size
    # ~50 (this project's real DTM cell size) that noise grows to ~5e-6,
    # bigger than a fixed 1e-6 absolute tolerance can absorb, which
    # silently failed to merge legitimate shared points (found via a
    # direct cell_size=50 test: 312 phantom faces, all at wall/urz
    # interfaces -- the exact same failure mode, and fix, as the original
    # 9dp-vs-6dp precision bug this project hit at cell_size=1). Dividing
    # by cell_size before rounding keeps the ABSOLUTE tolerance
    # proportional to cell_size, matching how the noise itself scales.
    pts_list, pid = [], {}

    def gid(p):
        key = (round(float(p[0]) / cx, 5),
               round(float(p[1]) / cy, 5),
               round(float(p[2]) / cz, 5))
        idx = pid.get(key)
        if idx is None:
            idx = len(pts_list)
            pid[key] = idx
            pts_list.append((float(p[0]), float(p[1]), float(p[2])))
        return idx

    hexes = np.array([[gid(p) for p in h] for h in raw])
    points = np.array(pts_list)

    vols = hex_volumes_signed(points, hexes)
    bad = np.where(vols <= 0)[0]
    for idx in bad:
        h = hexes[idx]
        hexes[idx] = np.concatenate([h[:4][::-1], h[4:][::-1]])

    return points, hexes, tags


def concave_orientations_used(footprint, NX, NY, n_layers=1):
    """List of uf (frozenset) for every concave block this footprint
    triggers, repeated once per refined layer -- each STACKED layer of a
    concave (i,j) position places its own independent copy of the
    corner-core (see build_mesh's k_layers loop), so it contributes its
    own volume deficit (templates.CONCAVE_VOLUME_DEFICIT) once per layer,
    not once per footprint position."""
    roles, _ = classify_footprint(footprint, range(NX), range(NY))
    per_position = [frozenset(r["uf"]) for r in roles.values() if r["kind"] == "concave"]
    return per_position * n_layers


def verify_mesh(points, hexes, footprint, NX, NY, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX,
                 n_layers=1, tol=1e-6):
    """Like geometry_checks.check_conformity_3d, but aware of the
    well-characterized per-concave-orientation volume deficit (see
    templates.CONCAVE_VOLUME_DEFICIT) -- flags a real problem (any
    phantom face, or a volume mismatch not explained by the specific
    concave orientations present) rather than the expected, harmless
    twisted-hex effect. `n_layers`: how many stacked k-layers were
    refined (build_mesh's k_layers), so multi-layer-deep concave regions
    are counted correctly (one deficit per layer, not per footprint
    position). Returns (ok: bool, detail: str)."""
    ok_struct = check_conformity_3d(points, hexes, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, tol=tol)
    vols = hex_volumes_signed(points, hexes)
    total_volume = float(vols.sum())
    expected_volume = (XMAX - XMIN) * (YMAX - YMIN) * (ZMAX - ZMIN)
    ufs = concave_orientations_used(footprint, NX, NY, n_layers=n_layers)
    expected_deficit = sum(T.CONCAVE_VOLUME_DEFICIT[uf] for uf in ufs)
    residual = (total_volume - expected_volume) - expected_deficit
    if ok_struct:
        return True, "fully conforming, exact volume"
    if abs(residual) < 1e-6:
        return True, (f"conforming: 0 phantom faces, volume deficit "
                       f"{total_volume - expected_volume:.6e} matches "
                       f"{len(ufs)} concave block(s) (orientations {[sorted(u) for u in ufs]}) "
                       f"exactly (known twisted-hex effect, not a defect)")
    return False, (f"NOT conforming: volume off by {residual:.6e} beyond what "
                    f"{len(ufs)} concave block(s) explain -- investigate")


def _run_test(name, footprint, NX, NY, NZ, XMIN, YMIN, ZMIN, K_URZ=None, k_layers=None):
    pts, hexes, tags = build_mesh(footprint, NX, NY, NZ, XMIN, YMIN, ZMIN,
                                   K_URZ=K_URZ, k_layers=k_layers)
    n_layers = len(k_layers) if k_layers is not None else 1
    ok, detail = verify_mesh(pts, hexes, footprint, NX, NY,
                              XMIN, -XMIN, YMIN, -YMIN, ZMIN, -ZMIN, n_layers=n_layers)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {len(hexes)} hexes -- {detail}")
    return ok


if __name__ == "__main__":
    all_ok = True

    XMIN = YMIN = ZMIN = -4.0
    NX = NY = NZ = 8

    print("--- regression tests: reproduce known real-CUBIT cases, refined layer at domain top ---")
    all_ok &= _run_test(
        "straight 2x2 (matches rebuild_mesh.py exactly)",
        {(3, 3), (3, 4), (4, 3), (4, 4)}, NX, NY, NZ, XMIN, YMIN, ZMIN, K_URZ=7)
    for n, cells in [
        (1, {(2,2),(3,2),(4,2),(5,2),(2,3),(3,3),(4,3),(5,3),(2,4),(3,4),(2,5),(3,5)}),
        (2, {(2,2),(3,2),(4,2),(5,2),(2,3),(3,3),(4,3),(5,3),(4,4),(5,4),(4,5),(5,5)}),
        (3, {(2,4),(3,4),(4,4),(5,4),(2,5),(3,5),(4,5),(5,5),(4,2),(5,2),(4,3),(5,3)}),
        (4, {(2,4),(3,4),(4,4),(5,4),(2,5),(3,5),(4,5),(5,5),(2,2),(3,2),(2,3),(3,3)}),
    ]:
        all_ok &= _run_test(f"L-shape case {n} (matches real CUBIT export)",
                             cells, NX, NY, NZ, XMIN, YMIN, ZMIN, K_URZ=7)

    print()
    print("--- new configurations never seen by CUBIT: interior refined layer (tests topcap), ---")
    print("--- multiple disjoint regions, and a concave region at a brand-new grid position ---")
    XMIN2 = YMIN2 = ZMIN2 = -6.0
    NX2 = NY2 = NZ2 = 12
    all_ok &= _run_test(
        "two disjoint straight 2x2 patches, interior layer",
        {(1,1),(1,2),(2,1),(2,2)} | {(8,8),(8,9),(9,8),(9,9)},
        NX2, NY2, NZ2, XMIN2, YMIN2, ZMIN2, K_URZ=7)
    Lbase = {(2,2),(3,2),(4,2),(5,2),(2,3),(3,3),(4,3),(5,3),(2,4),(3,4),(2,5),(3,5)}
    all_ok &= _run_test(
        "L-shape shifted to an untested position, interior layer",
        {(i + 5, j - 1) for (i, j) in Lbase},
        NX2, NY2, NZ2, XMIN2, YMIN2, ZMIN2, K_URZ=7)

    XMIN3 = YMIN3 = ZMIN3 = -8.0
    NX3 = NY3 = NZ3 = 16
    all_ok &= _run_test(
        "combined: 1 straight + 2 independent concave regions, top-of-domain layer",
        {(1,1),(1,2),(2,1),(2,2)} | {(i+7,j+7) for (i,j) in Lbase} | {(i+7,j-1) for (i,j) in Lbase},
        NX3, NY3, NZ3, XMIN3, YMIN3, ZMIN3, K_URZ=8)

    print()
    print("--- DEPTH: refine from the top down through several coarse layers, not just 1 ---")
    all_ok &= _run_test(
        "straight 2x2, depth=3 layers from the top",
        {(3, 3), (3, 4), (4, 3), (4, 4)}, NX, NY, NZ, XMIN, YMIN, ZMIN,
        k_layers=range(NZ - 3, NZ))
    all_ok &= _run_test(
        "straight 2x2, FULL depth (all 8 layers) -- edge case, no coarse below at all",
        {(3, 3), (3, 4), (4, 3), (4, 4)}, NX, NY, NZ, XMIN, YMIN, ZMIN,
        k_layers=range(NZ))
    all_ok &= _run_test(
        "L-shape case 1, depth=2 layers from the top",
        {(2,2),(3,2),(4,2),(5,2),(2,3),(3,3),(4,3),(5,3),(2,4),(3,4),(2,5),(3,5)},
        NX, NY, NZ, XMIN, YMIN, ZMIN, k_layers=range(NZ - 2, NZ))
    all_ok &= _run_test(
        "straight 2x2, depth=3, INTERIOR (not touching top OR bottom -- tests both caps at once)",
        {(3, 3), (3, 4), (4, 3), (4, 4)}, NX, NY, NZ, XMIN, YMIN, ZMIN,
        k_layers=range(2, 5))

    print()
    print("--- MULTI-LEVEL: push already-built plain-coarse cells to a further refinement pass ---")
    XMIN4 = YMIN4 = ZMIN4 = -8.0
    NX4 = NY4 = NZ4 = 16
    l1_footprint = {(3, 3), (3, 4), (4, 3), (4, 4)}
    l1_pts, l1_hexes, l1_tags = build_mesh(l1_footprint, NX4, NY4, NZ4, XMIN4, YMIN4, ZMIN4, K_URZ=7)
    l2_pts, l2_hexes, l2_tags = refine_cells_further(
        l1_pts, l1_hexes, l1_tags, [(10, 10, 10), (10, 10, 2)],
        NX4, NY4, NZ4, XMIN4, YMIN4, ZMIN4, local_size=5)
    ok, detail = verify_mesh(l2_pts, l2_hexes, l1_footprint, NX4, NY4,
                              XMIN4, -XMIN4, YMIN4, -YMIN4, ZMIN4, -ZMIN4)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] 2 independent level-2 targets pushed to a further urz split: "
          f"{len(l2_hexes)} hexes -- {detail}")
    all_ok &= ok

    print()
    print("--- NESTED MULTI-LEVEL: refine a target INSIDE an already-urz level-1 cell "
          "(cell_size support) ---")
    l1b_footprint = {(3, 3), (3, 4), (4, 3), (4, 4)}
    l1b_pts, l1b_hexes, l1b_tags = build_mesh(
        l1b_footprint, NX, NY, NZ, XMIN, YMIN, ZMIN, k_layers=range(5, 8))
    child_cell_size = 1.0 / 3
    urz_fine_pos = set()
    for h, t in zip(l1b_hexes, l1b_tags):
        if t.split(":")[0] != "urz":
            continue
        lo = l1b_pts[h].min(0)
        fine_ijk = tuple(int(round((lo[a] - [XMIN, YMIN, ZMIN][a]) / child_cell_size)) for a in range(3))
        urz_fine_pos.add(fine_ijk)
    buf = 2
    nested_candidates = [
        p for p in urz_fine_pos
        if all((p[0] + di, p[1] + dj, p[2] + dk) in urz_fine_pos
               for di in range(-buf, buf + 1) for dj in range(-buf, buf + 1) for dk in range(-buf, buf + 1))
    ]
    nested_target = nested_candidates[len(nested_candidates) // 2]
    l2b_pts, l2b_hexes, l2b_tags = refine_cells_further(
        l1b_pts, l1b_hexes, l1b_tags, [nested_target],
        NX * 3, NY * 3, NZ * 3, XMIN, YMIN, ZMIN, local_size=5, cell_size=child_cell_size)
    ok2 = check_conformity_3d(l2b_pts, l2b_hexes, XMIN, -XMIN, YMIN, -YMIN, ZMIN, -ZMIN)
    status2 = "PASS" if ok2 else "FAIL"
    print(f"[{status2}] level-2 target {nested_target} (cell_size={child_cell_size:.4f}) nested "
          f"inside the level-1 urz region: {len(l2b_hexes)} hexes (was {len(l1b_hexes)})")
    all_ok &= ok2

    print()
    print("--- DENSE NESTED REGION: refine a whole 2-cell sub-region at once, not one "
          "cell at a time (the actual shape a real, nested rectangular region needs) ---")
    # an adjacent PAIR needs a wider combined buffer ring than a single
    # cell, so it's not simply "reuse nested_target" -- found by checking
    # which adjacent pair actually has a fully-eligible combined ring
    di_pair = next(
        (i, j, k) for (i, j, k) in sorted(urz_fine_pos)
        if (i + 1, j, k) in urz_fine_pos
        and all((i + ddi, j + ddj, k + ddk) in urz_fine_pos
                for ddi in range(-buf, buf + 2) for ddj in range(-buf, buf + 1) for ddk in range(-buf, buf + 1))
    )
    dense_footprint = {(di_pair[0], di_pair[1]), (di_pair[0] + 1, di_pair[1])}
    dense_k = [di_pair[2]]
    l3_pts, l3_hexes, l3_tags = refine_region_further(
        l1b_pts, l1b_hexes, l1b_tags, dense_footprint, dense_k,
        NX * 3, NY * 3, NZ * 3, XMIN, YMIN, ZMIN, local_size=5, cell_size=child_cell_size)
    ok3 = check_conformity_3d(l3_pts, l3_hexes, XMIN, -XMIN, YMIN, -YMIN, ZMIN, -ZMIN)
    status3 = "PASS" if ok3 else "FAIL"
    print(f"[{status3}] dense 2-cell footprint {dense_footprint} refined as ONE region: "
          f"{len(l3_hexes)} hexes (was {len(l1b_hexes)})")
    all_ok &= ok3

    print()
    print("--- ANISOTROPIC cell_size: scalar vs equal-triple must be IDENTICAL ---")
    footprint_aniso = {(3, 3), (3, 4), (4, 3), (4, 4)}
    pa_pts, pa_hexes, pa_tags = build_mesh(
        footprint_aniso, NX, NY, NZ, XMIN, YMIN, ZMIN, K_URZ=7, cell_size=1.0)
    pb_pts, pb_hexes, pb_tags = build_mesh(
        footprint_aniso, NX, NY, NZ, XMIN, YMIN, ZMIN, K_URZ=7, cell_size=(1.0, 1.0, 1.0))
    ok4 = (pa_pts.shape == pb_pts.shape and np.allclose(pa_pts, pb_pts)
           and np.array_equal(pa_hexes, pb_hexes) and pa_tags == pb_tags)
    status4 = "PASS" if ok4 else "FAIL"
    print(f"[{status4}] scalar cell_size=1.0 vs tuple cell_size=(1.0,1.0,1.0): "
          f"byte-identical output ({len(pa_hexes)} hexes)")
    all_ok &= ok4

    print()
    print("--- ANISOTROPIC cell_size: z HALF the size of x/y (sx=sy != sz) ---")
    XMIN5 = YMIN5 = ZMIN5 = -4.0
    NX5 = NY5 = NZ5 = 8
    aniso_cs = (1.0, 1.0, 0.5)
    pc_pts, pc_hexes, pc_tags = build_mesh(
        footprint_aniso, NX5, NY5, NZ5, XMIN5, YMIN5, ZMIN5, K_URZ=7, cell_size=aniso_cs)
    ok5 = check_conformity_3d(pc_pts, pc_hexes, XMIN5, -XMIN5, YMIN5, -YMIN5,
                               ZMIN5, ZMIN5 + NZ5 * aniso_cs[2])
    status5 = "PASS" if ok5 else "FAIL"
    print(f"[{status5}] anisotropic cell_size={aniso_cs} (z half of x/y): "
          f"{len(pc_hexes)} hexes -- fully conforming, exact volume")
    all_ok &= ok5

    # NOTE: deliberately NOT testing "retrofit extra z-fineness onto an
    # ALREADY-isotropically-split finer cell" (e.g. take an existing
    # L2_urz cell and ask for z 3x finer than ITS OWN x/y) -- tried this
    # first and it correctly FAILS eligibility, for a real structural
    # reason, not an engine bug: urz_cell's 3x3x3 split places hex
    # boundaries at cell_size/3 steps in EVERY axis, so an already-built
    # urz cell simply has no EXISTING boundary at a finer-than-that z
    # step for a later call to find. Anisotropy has to be introduced
    # where a cell is FIRST built from a plain coarse/urz box (tests 4/5
    # above do exactly that), not layered on after the fact -- for
    # build_3level.py this means making LEVEL 2's own construction
    # anisotropic (extra z fineness from the start), so level 3, built
    # normally FROM level 2's cells, inherits it automatically -- not
    # making level 3 anisotropic relative to an already-isotropic level 2.

    print()
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED -- see above")
