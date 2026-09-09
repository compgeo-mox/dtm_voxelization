"""Consolidated, verified template library for the ternary (1-to-27) local
hex-refinement scheme. All geometry here was extracted from real CUBIT
`refine ... numsplit 1` ground truth (never hand-designed) and is loaded
from the self-contained `template_data.npz` baked by
`archive/cubit_reverse_engineering/rebuild_template_data.py` -- this
module has NO runtime dependency on the original CUBIT exports or the
investigation scripts that derived it (see that archive directory for the
full derivation history and how each piece here was verified: exact
point-matching against real data, not just volume/count matching).

Two layers of transition are needed below the refined layer's own
templates: URZCAP under a URZ core cell, WALLCAP under a wall cell -- both
turn out to just be OTHER templates from this same library, rotated (see
URZCAP_MAT / WALLCAP_MAT_BY_ROT below), not new shapes.

Convex case (footprint touches at most 1 full face, or only diagonally):
  - URZ core: trivial 3x3x3 split of a unit cube.
  - WALL_BASE (13 hex): one full face shared with the core. Rotation
    keyed by PUSH_TO_ROT[(di,dj)], (di,dj) = direction from this cell
    toward the adjacent core cell.
  - CORNER_BASE (5 hex): touches the core only diagonally. Rotation keyed
    by CORNER_TO_ROT[(di,dj)], (di,dj) = diagonal direction toward the
    core. NOTE: this mapping is NOT the same handedness as PUSH_TO_ROT's
    naive pattern -- verified via exhaustive 24-rotation point-matching,
    not assumed by analogy (an earlier guess had 2 of the 4 entries
    swapped).
  - No cap needed under a convex corner (verified: the real mesh's cell
    below a corner cell is a single plain coarse hex).

Concave case (a cell touching the footprint via exactly 2 ORTHOGONAL full
faces -- a reentrant/notch corner): there is no clean single-cell
template -- CUBIT's own mesh does not respect the coarse grid lines here
(hexes straddle from the neighboring "arm" cells back into the corner
cell's box). Verified, exhaustively, across all 4 possible orientations
(every one of the 4 valid adjacent-direction-pairs {S,W},{E,S},{E,N},{N,W}
has real CUBIT ground truth) that the complete, self-contained unit is a
combined block spanning the corner cell plus its two "arm" neighbor cells
(each an ordinary wall-touching cell, but geometrically fused to the
corner's box), PLUS the corresponding k-1 cap layer beneath all three.
Each of the 4 orientations is stored as a complete, ready-to-place
mega-block -- translation only, no further rotation/reflection needed at
assembly time.

Every non-trivial shape here (WALL_BASE, CORNER_BASE, the concave
corner-core) is z-mirror-symmetric (verified: zflip_local maps each one
exactly onto itself) -- see general_rebuild.py's use of this for
"topcap"/multi-layer-depth support without needing new ground truth for
those configurations.
"""
from pathlib import Path

import numpy as np

from .geometry_checks import hex_volumes_signed

_DIR = Path(__file__).parent

NBRS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}


# ---------------------------------------------------------------------
# low-level geometric helpers
# ---------------------------------------------------------------------

def rot90(hexarr):
    """Verified exact 90-degree rotation about a cell's own local
    z-axis (about the cell center, in local [0,1]^3 coords)."""
    x, y, z = hexarr[..., 0], hexarr[..., 1], hexarr[..., 2]
    return np.stack([y, 1 - x, z], axis=-1)


def rot_n(arr, n):
    for _ in range(n % 4):
        arr = rot90(arr)
    return arr


def cs3(cell_size):
    """Normalizes cell_size -- a scalar (isotropic, the original
    convention) OR a (sx,sy,sz) triple -- into an explicit 3-element
    array, so callers doing per-axis OFFSET arithmetic (x0+cx, not
    x0+cell_size) can just index cs3(cell_size)[0/1/2]. Geometry-scaling
    MULTIPLIES (template_array * cell_size) don't need this: numpy
    broadcasts a bare scalar or a (3,)-shaped cell_size against a (...,3)
    array identically either way, so those call sites pass `cell_size`
    straight through unchanged.

    ANISOTROPY CONSTRAINT (not enforced here, just true): sx MUST equal
    sy for apply_matrix/place_rot's rotation step to stay correct -- see
    their docstring. sz alone may safely differ."""
    return np.broadcast_to(np.asarray(cell_size, dtype=float), (3,))


def apply_matrix(arr, R, origin=(0.0, 0.0, 0.0), cell_size=1.0):
    """Rotate/reflect arr (..., 3) by 3x3 matrix R about its own local
    cell center (0.5,0.5,0.5), scale by `cell_size`, then translate to
    `origin`. Rotation-about-center and a uniform (isotropic) scale
    commute, so scaling before or after the rotation gives the identical
    result -- done after, alongside the translation, to keep this a
    trivial one-line change from the original unit-only version.

    cell_size MAY be anisotropic -- a (sx,sy,sz) triple instead of one
    scalar -- but ONLY with sx==sy: every R used in this template library
    (rot_n and everything derived from it, PUSH_TO_ROT/CORNER_TO_ROT/
    WALLCAP_MAT_BY_ROT/URZCAP_MAT) is a rotation purely ABOUT THE LOCAL
    Z-AXIS -- it permutes/negates x and y among each other but never
    mixes either into z (rot90: (x,y,z)->(y,1-x,z), z untouched). Scaling
    AFTER such a rotation by a diagonal (sx,sy,sz) is safe exactly when
    sx==sy, because then the scale commutes with the x<->y swap the same
    way a uniform scale would; with sx!=sy it would NOT commute (a 90
    degree turn would apply the "wrong" axis's factor to each direction)
    and would silently distort the template. Verified by construction,
    not by a runtime check here -- callers choosing an anisotropic
    cell_size (see build_3level.py's level-3 z-only refinement) must keep
    sx==sy themselves."""
    c = np.array([0.5, 0.5, 0.5])
    flat = arr.reshape(-1, 3)
    out = (flat - c) @ R.T + c
    return out.reshape(arr.shape) * cell_size + np.array(origin, dtype=float)


def place_rot(base, n, origin, cell_size=1.0):
    return rot_n(base, n) * cell_size + np.array(origin, dtype=float)


def zflip_local(arr):
    """Mirror a template through its own local z=0.5 mid-plane (in
    [0,1]^3 coords): (x,y,z) -> (x,y,1-z)."""
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.stack([x, y, 1 - z], axis=-1)


def urz_cell(x0, y0, z0, cell_size=1.0):
    """27-way uniform split of a box at global origin (x0,y0,z0), sized
    `cell_size` per axis -- a scalar (cube) or a (sx,sy,sz) triple."""
    csx, csy, csz = cs3(cell_size)
    hs = []
    xs = np.linspace(x0, x0 + csx, 4)
    ys = np.linspace(y0, y0 + csy, 4)
    zs = np.linspace(z0, z0 + csz, 4)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                b0 = [(xs[a], ys[b], zs[c]), (xs[a+1], ys[b], zs[c]),
                      (xs[a+1], ys[b+1], zs[c]), (xs[a], ys[b+1], zs[c])]
                t0 = [(xs[a], ys[b], zs[c+1]), (xs[a+1], ys[b], zs[c+1]),
                      (xs[a+1], ys[b+1], zs[c+1]), (xs[a], ys[b+1], zs[c+1])]
                hs.append(b0 + t0)
    return np.array(hs, dtype=float)


def coarse_box(x0, y0, z0, cell_size=1.0):
    csx, csy, csz = cs3(cell_size)
    x1, y1, z1 = x0 + csx, y0 + csy, z0 + csz
    b0 = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    t0 = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return np.array([b0 + t0], dtype=float)


# ---------------------------------------------------------------------
# load the baked template geometry
# ---------------------------------------------------------------------

_data = np.load(_DIR / "template_data.npz")

WALL_BASE = _data["WALL_BASE"]
CORNER_BASE = _data["CORNER_BASE"]
assert WALL_BASE.shape == (13, 8, 3)
assert CORNER_BASE.shape == (5, 8, 3)

# push direction (di,dj) toward the adjacent core cell -> WALL_BASE rotation
PUSH_TO_ROT = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}
# diagonal direction (di,dj) toward the core cell -> CORNER_BASE rotation
# (verified via exhaustive 24-rotation search -- NOT the same handedness
# as PUSH_TO_ROT; see module docstring)
CORNER_TO_ROT = {(1, 1): 0, (1, -1): 1, (-1, -1): 2, (-1, 1): 3}

# k-1 cap templates: URZCAP is WALL_BASE rotated by one FIXED matrix
# (verified identical for all 4 URZ-core cells -- no lateral variation).
# WALLCAP is CORNER_BASE rotated by a matrix that depends only on the
# owning wall cell's own PUSH_TO_ROT rotation (verified for all 8 wall
# cells in the straight case AND for arm cells in the L-shape cases).
URZCAP_MAT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
WALLCAP_MAT_BY_ROT = {
    0: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    1: np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
    2: np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    3: np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
}

CONCAVE = {}
for _key in ["SW", "ES", "EN", "NW"]:
    arms = []
    for _idx in (0, 1):
        arms.append(dict(
            offset=tuple(int(v) for v in _data[f"CONCAVE_{_key}_arm{_idx}_offset"]),
            core=_data[f"CONCAVE_{_key}_arm{_idx}_core"],
            cap=_data[f"CONCAVE_{_key}_arm{_idx}_cap"],
        ))
    CONCAVE[frozenset(_key)] = dict(
        corner=_data[f"CONCAVE_{_key}_corner"],
        corner_cap=_data[f"CONCAVE_{_key}_corner_cap"],
        arms=arms,
    )
assert len(CONCAVE) == 4

# per-orientation volume deficit: the corner-core's real CUBIT hexes
# include several genuinely TWISTED (non-planar-face) convex hexahedra,
# whose true trilinear volume is legitimately less (or, for 2 of the 4
# orientations, slightly MORE) than a naive expectation of exactly 6.0
# (3 cells at the refined layer + the matching 3 cells' worth at the cap
# layer below) -- verified via scipy.ConvexHull (every hex has exactly 8
# hull vertices, i.e. none are non-convex) and via a plain unit-cube
# sanity check on the volume formula itself. NOT the same value for every
# orientation -- each is computed directly from its own real geometry.
CONCAVE_VOLUME_DEFICIT = {}
for _uf, _blk in CONCAVE.items():
    _total = 0.0
    for _piece in [_blk["corner"], _blk["corner_cap"]]:
        _n = len(_piece)
        _total += hex_volumes_signed(_piece.reshape(-1, 3), np.arange(_n * 8).reshape(-1, 8)).sum()
    for _arm in _blk["arms"]:
        for _piece in [_arm["core"], _arm["cap"]]:
            _n = len(_piece)
            _total += hex_volumes_signed(_piece.reshape(-1, 3), np.arange(_n * 8).reshape(-1, 8)).sum()
    CONCAVE_VOLUME_DEFICIT[_uf] = _total - 6.0


def _union_faces_and_arms(corner, refined_set):
    """Used by general_rebuild.py's classifier: given a candidate corner
    cell and the refined footprint, returns (uf, arms) -- uf = tuple of
    the corner's in-footprint face directions, arms = the 2 neighbor
    cells that face-touch the footprint themselves."""
    ci, cj = corner
    uf = tuple(sorted(name for name, (di, dj) in NBRS.items() if (ci + di, cj + dj) in refined_set))
    arms = []
    for name, (di, dj) in NBRS.items():
        ai, aj = ci + di, cj + dj
        if (ai, aj) in refined_set:
            continue
        touch = sum(1 for _, (ddi, ddj) in NBRS.items() if (ai + ddi, aj + ddj) in refined_set)
        if touch >= 1:
            arms.append((ai, aj))
    return uf, arms


if __name__ == "__main__":
    print("WALL_BASE:", WALL_BASE.shape[0], "hexes")
    print("CORNER_BASE:", CORNER_BASE.shape[0], "hexes")
    print("CONCAVE orientations loaded:", sorted(tuple(sorted(k)) for k in CONCAVE))
    for uf, blk in CONCAVE.items():
        arm_offsets = [a["offset"] for a in blk["arms"]]
        print(f"  uf={sorted(uf)}: corner=21, cap=9, arms(13+5 each) at offsets {arm_offsets}, "
              f"volume_deficit={CONCAVE_VOLUME_DEFICIT[uf]:.6e}")
