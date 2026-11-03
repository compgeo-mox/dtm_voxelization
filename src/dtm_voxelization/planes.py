"""The two surveyed planes through the mountain, in the grid's frame.

The first comes as an equation, 0.2230 X + 0.4732 Y - 0.8523 Z + d = 0, fitted
in Monte Mario / Italy zone 1 -- see surveyed() for the journey its x and y
make and why its height has to be fitted rather than read. The second runs
DROPS below it. section.py draws them and cuts them with the vertical plane;
speed.py labels the rock above, between and below them.
"""

import numpy as np
from scipy.spatial import cKDTree

import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .dtm_io import _load_xyz, load_dtm_interpolator
from .shift_fracture import M, S  # the same Rialba mean and software shift

# a surveyed plane, as 0.2230 X + 0.4732 Y - 0.8523 Z + d = 0 with d / -0.8523 given,
# and a second one DROPS below it
PLANE = (0.2230, 0.4732, -0.8523, 3221403.4962)
# the SPEED scripts' Gauss-Boaga shift for the full Rialba (convert_stl_vtu.py,
# topograpy_regions.py), which is where the grid's origin sits in Gauss-Boaga
GAUSS_BOAGA = np.array([1527863.3, 5082212.3])
# the rock in five, first, as a .mate is written: SPEED reads quad and hex
# labels out of one column, and speed.py carries the boundary's two on from
# here, so the seven run 1 to 7 with nothing repeated and no hole
ROCK_TAG = 1  # what none of the three below claims
BETWEEN_TAG = 2  # the layer the planes hold, under the deposit and the conglomerate both
CONGLOMERATE_TAG = 3  # over the upper plane, inside the conglomerate, deposit or not
DEPOSIT_TAG = 4  # over the upper plane, inside the deposit, outside the conglomerate
BACK_TAG = 5  # the deposit behind the conglomerate: a skin, and only at the top

BACK_DEPTH = 5.0  # m of ground the back deposit takes, measured down from the terrain,
# or one hex where the hexes are taller than that -- see back_deposit()

# the two planes, below the one the fit leaves: the upper one 5 m under it, by
# hand and after looking at the grid, the lower one where it always was. They
# stand 55 m apart now, not 60
DROPS = (5.0, 60.0)
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


def outline_of(path, what):
    """The footprint the file draws, as a polygon in the grid's frame.

    The file is X Y Z in UTM32N and the z is of no use: what it draws is a
    footprint. The ring closes itself, so a file that repeats its first corner
    and one that does not both work, and buffer(0) settles whatever the
    digitizing left crossing itself."""
    if path is None:
        return None
    corners = np.loadtxt(path, skiprows=1, usecols=(0, 1)) - M[:2]
    shape = Polygon(corners).buffer(0)
    print(
        f"  {what} from {path.name}: {len(corners)} corners, {shape.area:,.0f} m2, x "
        f"[{corners[:, 0].min():.1f},{corners[:, 0].max():.1f}] y "
        f"[{corners[:, 1].min():.1f},{corners[:, 1].max():.1f}]",
        flush=True,
    )
    return shape


def seam_between(one, other):
    """The splinters two footprints leave between them: the holes of their
    union, which is what a contact digitized twice comes out as."""
    union = unary_union([one, other])
    parts = union.geoms if union.geom_type == "MultiPolygon" else [union]
    return unary_union([Polygon(part.exterior) for part in parts]).difference(union)


def sewn(deposit, conglomerate):
    """The two footprints with the slivers between them closed.

    They were drawn one at a time along the same contact, so between them they
    leave splinters -- at Rialba 15 of them, 176 m2 in all and none wider than
    2 m, which is thinner than a cell. A hex whose centre lands in one belongs
    to neither body and comes out labelled as plain rock, a blue seam along the
    contact. The splinters are the holes of the union, and they go to the
    conglomerate; no outer boundary moves."""
    if deposit is None or conglomerate is None:
        return deposit, conglomerate
    seam = seam_between(deposit, conglomerate)
    if seam.is_empty:
        return deposit, conglomerate
    splinters = seam.geoms if seam.geom_type == "MultiPolygon" else [seam]
    print(
        f"  the two meet along a contact that leaves {len(list(splinters))} splinters "
        f"between them, {seam.area:.1f} m2 in all: sewn onto the conglomerate",
        flush=True,
    )
    return deposit, unary_union([conglomerate, seam])


def footprint(shape, points, what):
    """Which of the points stand inside the shape. No shape, no limit."""
    if shape is None:
        print(f"  no {what} outline: the whole mesh counts as inside", flush=True)
        return np.ones(len(points), dtype=bool)
    shapely.prepare(shape)
    return shapely.contains_xy(shape, points[:, 0], points[:, 1])


def reached(middles, start, pool, reach):
    """Which of the pool's cells the start touches, stepping one cell at a
    time: the pool is walked outwards from the start until nothing more joins.

    `reach` is a cell's own diagonal in plan, so a step goes to a neighbour
    and no further: a body of pool cells cut off from the start by cells that
    are not in the pool stays out, which is the point of walking at all."""
    tree = cKDTree(middles[pool])
    taken = np.zeros(len(pool), dtype=bool)
    front, rounds = middles[start], 0
    while len(front):
        near = np.unique(np.concatenate(tree.query_ball_point(front, reach))).astype(int)
        fresh = near[~taken[near]]
        if not len(fresh):
            break
        taken[fresh] = True
        front, rounds = middles[pool[fresh]], rounds + 1
    print(
        f"    {int(taken.sum()):,} of {len(pool):,} of them joined it, in {rounds} round(s) "
        f"of {reach:.1f} m",
        flush=True,
    )
    return taken


def back_deposit(case, points, hexes, middles, tags, conglomerate, in_conglomerate):
    """The deposit behind the conglomerate laid over the tags, BACK_DEPTH deep.

    It is a skin, not a body: a hex takes BACK_TAG when its centre stands
    within BACK_DEPTH of the terrain over it and inside the drawn outline,
    which is why the terrain comes in here and nowhere else in this file.

    BACK_DEPTH is what is wanted, one hex is what is left when the hexes are
    taller than that -- a mesh cannot carry a layer thinner than a cell. So the
    reach is BACK_DEPTH or the hex's own height, whichever is more: where the
    hexes are the coarse ones the layer is the top hex of the column alone and
    nothing under it, and where they are finer than BACK_DEPTH the same rule
    comes down to the 5 m asked for, with no number to change here. Rialba's
    case refines that ground to 4.8 m at 2.5M cells, so it does; left coarse
    the hexes there would be 37 m. The log says how thick it came out.

    It takes the rock and the layer between the planes, blue and light blue,
    but never the conglomerate's own top: where the outline runs over the
    conglomerate -- the wedge whose ground still lies under the upper plane --
    a hex already labelled CONGLOMERATE_TAG keeps that label.

    Cutting the outline on the ground and the hexes on their centres leaves a
    strip between the two, hexes whose centre is under the upper plane though
    the ground over them is not: they stand between the new region and the
    conglomerate's top and go to the new region, which is what the walk from
    it through the skin picks up."""
    back = outline_of(case.back_deposit, "back deposit")
    if back is None:
        return tags
    if conglomerate is not None:
        seam = seam_between(back, conglomerate)
        if seam.area > 1.0:  # under a square metre is the shared boundary's own dust
            print(f"    they leave {seam.area:.1f} m2 of splinters between them, "
                  f"which go to the back deposit too", flush=True)
            back = unary_union([back, seam])

    ground = load_dtm_interpolator(case.points)(middles[:, :2])
    corners_z = points[:, 2][hexes]  # (n, 8), not the whole (n, 8, 3): this runs on 2.5M hexes
    heights = corners_z.max(axis=1) - corners_z.min(axis=1)
    # how deep the hex's TOP stands under the terrain, not its centre. The
    # carving keeps a hex when its top is under the ground over its middle
    # (terrain.keep), so the top hex of a column carries its centre as much as
    # one and a half cells down -- 7.3 m where the cells are 4.8 -- and a depth
    # read there misses it: at Rialba it left 43 per cent of the columns with
    # no layer at all. Read at the top, a hex is in when it reaches into the
    # top BACK_DEPTH metres of ground, which is what a layer that thick is, and
    # the top hex of a column is in whatever its size.
    skin = ground - corners_z.max(axis=1) < np.maximum(BACK_DEPTH, heights)
    in_back = footprint(back, middles[:, :2], "back deposit")
    take = skin & in_back & (tags != CONGLOMERATE_TAG)
    print(
        f"  the back deposit: {int(in_back.sum()):,} hexes under its outline, "
        f"{int(take.sum()):,} of them in the top {BACK_DEPTH:g} m of the terrain, or in "
        f"the top hex where a hex is taller than that, and not the conglomerate's own top "
        f"({int((skin & in_back & (tags == CONGLOMERATE_TAG)).sum()):,} left to it)",
        flush=True,
    )

    gap = skin & (in_back | in_conglomerate) & (tags != CONGLOMERATE_TAG) & ~take
    if gap.any() and take.any():
        # a step is as long as the widest cell in play, not the widest in the
        # mesh: the walk must reach a neighbour here and stop there
        corners = points[hexes[take | gap]]
        sides = corners.max(axis=1) - corners.min(axis=1)
        reach = 1.01 * np.hypot(sides[:, 0], sides[:, 1]).max()
        pool = np.where(gap)[0]
        print(f"    {len(pool):,} hexes of the skin lie between it and the conglomerate's top:",
              flush=True)
        take[pool[reached(middles, np.where(take)[0], pool, reach)]] = True

    if take.any():
        thick = heights[take]
        print(
            f"    the layer comes out {np.median(thick):.1f} m thick at the median, "
            f"{thick.min():.1f} to {thick.max():.1f}, against the {BACK_DEPTH:g} m asked for",
            flush=True,
        )
    for tag in (ROCK_TAG, BETWEEN_TAG, DEPOSIT_TAG):
        n = int((take & (tags == tag)).sum())
        if n:
            print(f"    taken off tag {tag}: {n:,} hexes", flush=True)
    tags[take] = BACK_TAG
    return tags


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

    Last comes the deposit behind the conglomerate, BACK_TAG, which is not a
    body between the planes but a skin BACK_DEPTH deep under the terrain, laid
    over whatever the rules above left there -- see back_deposit().

    A hex goes by its centre, so the planes and the outlines cut along cell
    faces -- they are not conformed to, which at a metre of cell is a metre of
    error on surfaces known to far less than that.

    The planes and their fit are Rialba's own, and so are the outlines that
    say where they hold: a case carrying neither is one body of one rock, and
    gets ROCK_TAG throughout without any of this being asked of it."""
    if case.deposit is None and case.conglomerate is None:
        print(
            f"  no deposit or conglomerate outline: one material, tag {ROCK_TAG}, "
            f"over all {len(hexes):,} hexes",
            flush=True,
        )
        return np.full(len(hexes), ROCK_TAG)

    middles = points[hexes].mean(axis=1)
    sink = under_the_slope(case)
    upper = surveyed(middles[:, :2], sink + DROPS[0])
    lower = surveyed(middles[:, :2], sink + DROPS[1])
    over = middles[:, 2] > upper
    deposit, conglomerate = sewn(
        outline_of(case.deposit, "deposit"),
        outline_of(case.conglomerate, "conglomerate"),
    )
    in_deposit = footprint(deposit, middles[:, :2], "deposit")
    in_conglomerate = footprint(conglomerate, middles[:, :2], "conglomerate")

    tags = np.full(len(hexes), ROCK_TAG)
    tags[(in_deposit | in_conglomerate) & ~over & (middles[:, 2] >= lower)] = BETWEEN_TAG
    tags[in_deposit & over] = DEPOSIT_TAG
    tags[in_conglomerate & over] = CONGLOMERATE_TAG
    tags = back_deposit(
        case, points, hexes, middles, tags, conglomerate, in_conglomerate
    )
    for tag, label in [
        (ROCK_TAG, "outside both outlines, or under the lower plane"),
        (BETWEEN_TAG, "between the two planes, under either body"),
        (CONGLOMERATE_TAG, "the conglomerate, over the upper plane"),
        (DEPOSIT_TAG, "the deposit, over the upper plane"),
        (BACK_TAG, f"behind the conglomerate, {BACK_DEPTH:g} m of ground or one hex"),
    ]:
        print(f"  tag {tag} ({label}): {int((tags == tag).sum()):,} hexes", flush=True)
    return tags
