"""The two surveyed planes through the mountain, in the grid's frame.

The first comes as an equation, 0.2230 X + 0.4732 Y - 0.8523 Z + d = 0, fitted
in Monte Mario / Italy zone 1 -- see surveyed() for the journey its x and y
make and why its height has to be fitted rather than read. The second runs
DROPS below it. section.py draws them and cuts them with the vertical plane;
speed.py labels the rock above, between and below them.
"""

import numpy as np
from scipy.spatial import cKDTree

from .dtm_io import _load_xyz
from .fracture_surface import inside
from .shift_fracture import M, S  # the same Rialba mean and software shift

# a surveyed plane, as 0.2230 X + 0.4732 Y - 0.8523 Z + d = 0 with d / -0.8523 given,
# and a second one DROPS below it
PLANE = (0.2230, 0.4732, -0.8523, 3221403.4962)
# the SPEED scripts' Gauss-Boaga shift for the full Rialba (convert_stl_vtu.py,
# topograpy_regions.py), which is where the grid's origin sits in Gauss-Boaga
GAUSS_BOAGA = np.array([1527863.3, 5082212.3])
# the rock in four, first, as a .mate is written: SPEED reads quad and hex
# labels out of one column, and speed.py carries the boundary's two on from
# here, so the six run 1 to 6 with nothing repeated and no hole
ROCK_TAG = 1  # what none of the three below claims
BETWEEN_TAG = 2  # the layer the planes hold, under the deposit and the conglomerate both
CONGLOMERATE_TAG = 3  # over the upper plane, inside the conglomerate, deposit or not
DEPOSIT_TAG = 4  # over the upper plane, inside the deposit, outside the conglomerate
CHUNK = 200_000  # hexes per batch of the point-in-polygon test, to bound its matrix

DROPS = (0.0, 60.0)  # the planes to draw, below the one the fit leaves
UNDER = 10.0  # m the first plane is wanted under the slope that runs parallel to it
PARALLEL = 12.0  # degrees within which the ground counts as parallel to the plane
NEIGHBOURS = 10  # ground points a local normal is taken from
FIT_BOX = (-550.0, -160.0, 20.0, 160.0)  # where to fit: the slope below the towers


def under_the_slope(case):
    """How far the equation's plane has to come down to run UNDER m below the
    ground, measured where the ground lies parallel to it.

    The constant of the equation is given to five digits and its last one is
    worth 100 m of elevation, so the plane's height is the one thing it does
    not fix: as it stands the plane comes out of the slope below the towers.
    What it does fix is the attitude, so the plane is dropped until it runs
    UNDER metres below that slope -- the one piece of ground that lies parallel
    to it, where a constant distance is a thing a plane can hold.

    Only the ground inside FIT_BOX counts, and of it only the points whose own
    normal, taken from their NEIGHBOURS nearest, stands within PARALLEL degrees
    of the plane's. The median gap is what is matched; a slope is rough, so
    the mean would follow whatever gullies are in the box."""
    x, y, z = _load_xyz(case.points)
    ground = np.column_stack([x, y, z])
    x0, y0, x1, y1 = FIT_BOX
    inside_box = (
        (ground[:, 0] > x0) & (ground[:, 0] < x1) & (ground[:, 1] > y0) & (ground[:, 1] < y1)
    )
    ground = ground[inside_box]
    if len(ground) < NEIGHBOURS:
        raise SystemExit(f"the fit box {FIT_BOX} holds {len(ground)} ground points")

    _, near = cKDTree(ground[:, :2]).query(ground[:, :2], k=NEIGHBOURS)
    patch = ground[near] - ground[near].mean(axis=1)[:, None, :]
    normals = np.linalg.svd(patch, full_matrices=False)[2][:, 2, :]
    a, b, c = PLANE[:3]
    upright = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    flat = np.degrees(np.arccos(np.clip(np.abs(normals @ upright), 0, 1))) < PARALLEL
    if not flat.any():
        raise SystemExit(
            f"no ground in {FIT_BOX} lies within {PARALLEL} degrees of the plane"
        )

    gap = ground[flat, 2] - surveyed(ground[flat, :2], 0.0)
    drop = UNDER - np.median(gap)
    left = gap + drop
    print(
        f"  the plane's height, fitted on {flat.sum():,} of {len(ground):,} ground points in "
        f"{FIT_BOX} whose slope runs within {PARALLEL:g} degrees of it:\n"
        f"    it stands {-np.median(gap):.1f} m over them, wanted {UNDER:g} m under, "
        f"so it comes down {drop:.1f} m\n"
        f"    after which the ground is {np.median(left):.1f} m above it at the median, "
        f"{np.percentile(left, 25):.1f} to {np.percentile(left, 75):.1f} between the quartiles, "
        f"and still below it at {100 * (left < 0).mean():.0f} per cent of them"
    )
    return drop


def surveyed(points, drop):
    """The surveyed plane's z, DROP metres lower, over the given x and y.

    The plane was fitted in Monte Mario / Italy zone 1 (Gauss-Boaga), whose
    easting runs a million metres ahead of UTM's: read in UTM the equation puts
    the plane 262 km underground, that million times 0.2230 / 0.8523. The shift
    that carries the grid there is the SPEED scripts' own for the full Rialba,
    the one turning frattura_verticale.stl into the shifted copy this repo
    carries; reprojecting the origin properly lands 2.5 m from it in x and 3 cm
    in y, which is 0.7 m of z and not worth a dependency.

    Only, the data the plane was fitted on had been shifted by the software's
    own S first and that Gauss-Boaga shift put back on top -- the two are mixed
    in the SPEED scripts' merge_dtm.py, under a comment that already doubts
    them -- so M + S goes on again here. It is worth 0.2230 / 0.8523 times
    837 m of easting plus 0.4732 / 0.8523 times 191 m of northing, 325 m of
    elevation: without it the plane runs 399 m under the terrain everywhere,
    with it 73 m under on average and crossing it, which is a plane a mountain
    can have.

    Its Z is an elevation above the sea, so the cloud's mean comes off it as it
    does off everything else. That is how the whole Gauss-Boaga lineage of this
    case is written -- frattura_verticale.stl, Rialba_DTM2x2.txt,
    DTMRialba5m_4Cubit -- where x and y are shifted and z never is.

    DROP comes off last, and is where under_the_slope() puts the plane."""
    a, b, c, offset = PLANE
    here = points + GAUSS_BOAGA + (M[:2] + S[:2])
    return -(a / c * here[:, 0] + b / c * here[:, 1] + offset) - M[2] - drop


def within(outline, points):
    """Which of the points fall inside the outline, a batch at a time.

    The crossing test weighs one row per point against one column per edge, so
    on a mesh of millions it is the matrix, not the work, that is the problem:
    87 edges of conglomerate against 2.5 M hexes is 1.7 GB of float in one
    go."""
    return np.concatenate(
        [inside(outline, points[at : at + CHUNK]) for at in range(0, len(points), CHUNK)]
    )


def footprint(path, points, what):
    """Which of the points stand inside the outline the file draws.

    The file is X Y Z in UTM32N and the z is of no use: what it draws is a
    footprint. The ring closes itself, so a file that repeats its first corner
    and one that does not both work. No file means no limit: everything is
    inside."""
    if path is None:
        print(f"  no {what} outline: the whole mesh counts as inside", flush=True)
        return np.ones(len(points), dtype=bool)
    outline = np.loadtxt(path, skiprows=1, usecols=(0, 1)) - M[:2]
    if np.allclose(outline[0], outline[-1]):
        outline = outline[:-1]
    print(
        f"  {what} from {path.name}: {len(outline)} corners, x "
        f"[{outline[:, 0].min():.1f},{outline[:, 0].max():.1f}] y "
        f"[{outline[:, 1].min():.1f},{outline[:, 1].max():.1f}]",
        flush=True,
    )
    return within(outline, points)


def materials(case, points, hexes):
    """Each hex labelled by where it falls against the two planes and the two
    outlines the case carries.

Both bodies stand on the upper plane: the conglomerate takes
    CONGLOMERATE_TAG over it and the deposit DEPOSIT_TAG, each inside its own
    outline. The two lie side by side, sharing a stretch of boundary and
    overlapping over 1,245 m2 only, so neither is tied to the other.

    BETWEEN_TAG is then the layer the planes hold between them, under either of
    the two -- it runs on under the conglomerate as it does under the deposit.
    Anywhere else, and under the lower plane, the rock is ROCK_TAG. An outline
    the case leaves out is taken as covering everything.

    A hex goes by its centre, so the planes and the outlines cut along cell
    faces -- they are not conformed to, which at a metre of cell is a metre of
    error on surfaces known to far less than that."""
    middles = points[hexes].mean(axis=1)
    sink = under_the_slope(case)
    upper = surveyed(middles[:, :2], sink + DROPS[0])
    lower = surveyed(middles[:, :2], sink + DROPS[1])
    over = middles[:, 2] > upper
    in_deposit = footprint(case.deposit, middles[:, :2], "deposit")
    in_conglomerate = footprint(case.conglomerate, middles[:, :2], "conglomerate")

    tags = np.full(len(hexes), ROCK_TAG)
    tags[(in_deposit | in_conglomerate) & ~over & (middles[:, 2] >= lower)] = BETWEEN_TAG
    tags[in_deposit & over] = DEPOSIT_TAG
    tags[in_conglomerate & over] = CONGLOMERATE_TAG
    for tag, label in [
        (ROCK_TAG, "outside both outlines, or under the lower plane"),
        (BETWEEN_TAG, "between the two planes, under either body"),
        (CONGLOMERATE_TAG, "the conglomerate, over the upper plane"),
        (DEPOSIT_TAG, "the deposit, over the upper plane"),
    ]:
        print(f"  tag {tag} ({label}): {int((tags == tag).sum()):,} hexes", flush=True)
    return tags
