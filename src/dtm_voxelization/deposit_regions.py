"""The two deposit outlines Rialba's case reads, made from what they are drawn
on: `python -m dtm_voxelization.deposit_regions [CASE.toml]`, or the file
itself from an editor, which takes CASE below.

They are written where the case says they are -- [deposit] and [back_deposit]
-- so running this is what keeps those files and this reasoning together. The
survey's own `deposito.txt` and the road's kmz are the inputs and are never
written. Beside them go the curves to check in ParaView, draped DRAPE metres
over the terrain so they do not sink into it: the two outlines and the road.

The extended deposit is the surveyed one carried south onto the SS36. Only the
arc that faces the road moves: the outline is spliced onto the asphalt between
two of its own vertices (ARC) and left alone everywhere else, which is what
the union of the two comes out as. The survey's ring crosses itself at four
spikes, and shapely drops those in the cleaning -- they are the only vertices
outside the arc that do not survive.

The back deposit is the ground behind the conglomerate, from its own back
boundary to the end of y. West it runs straight north off the conglomerate's
west extreme; east it follows EAST, read off the line drawn over the grid in
ParaView, which keeps clear of the blocks of cells climbing the slope there --
that line is a judgement, not a feature of the terrain: it stands 70 to 100 m
west of the foot of the slope and 13 m west of the 180 m contour. Between the
two feet the outline closes over the conglomerate's own digitized vertices,
except along the wedge where the ground inside the conglomerate still lies
under the upper plane, which is the material 2 this region was asked for in
the first place: there it leaves the outline and runs on that plane's outcrop,
so the wedge comes in.
"""

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import meshio
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, split, substring, unary_union

if __package__:
    from . import planes
    from .case import load_case
    from .dtm_io import load_dtm_interpolator
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization import planes
    from dtm_voxelization.case import load_case
    from dtm_voxelization.dtm_io import load_dtm_interpolator

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button
SURVEYED = "data/regions/deposito.txt"  # the deposit as it was drawn, the input here
ROAD = "data/xyz/SS36_Highway_Rialba.kmz"  # the SS36, one line string in degrees

# the arc of the surveyed ring the road replaces, indices on it once shapely
# has resolved the four self-crossing spikes it carries: from the deposit's
# south-east corner (-32.7, -124.1) to where its west side already runs 12 m
# off the asphalt (-495.3, 91.1). Picked on the plan view.
ARC = (92, 108)

# the back deposit's east boundary, one x every 10 m of y, in the grid's frame
EAST = np.array([
    [58.0, 525.0], [55.3, 535.0], [51.9, 545.0], [50.2, 555.0], [48.8, 565.0],
    [44.8, 575.0], [39.7, 585.0], [33.6, 595.0], [29.0, 605.0], [24.5, 615.0],
    [22.5, 625.0], [21.6, 635.0], [21.4, 645.0], [22.0, 655.0], [21.2, 665.0],
    [20.3, 675.0], [19.9, 685.0], [18.8, 695.0], [17.3, 705.0], [16.5, 715.0],
    [17.4, 725.0], [18.9, 735.0], [19.2, 745.0], [17.6, 755.0], [16.4, 765.0],
    [15.9, 775.0], [15.4, 785.0], [15.4, 795.0], [13.9, 805.0], [13.2, 815.0],
])
# the outline's north edge. The grid rounds its own north edge up to a whole
# number of cells, past the DTM's 805.4 -- 816.3 at 2.5M cells -- and the last
# row of hexes has its centres past 810, where this used to stop: they came
# out as rock, one row of them, along the whole end of the domain. It only has
# to sit past whatever the build rounds to, and there are no hexes out here.
YMAX = 900.0
# marching squares: which edges of a cell the zero level joins, by the corners
# that are negative. 0 is the edge between corner 0 (x, y) and corner 1
# (x + h, y), then round the cell. The two saddles, 5 and 10, are cut the way
# that keeps the two negative corners apart, which here only ever happens
# where the ground grazes the plane and either reading is the same curve.
JOINS = {1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 5: ((3, 0), (1, 2)),
         6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),), 9: ((2, 0),), 10: ((0, 1), (2, 3)),
         11: ((2, 1),), 12: ((1, 3),), 13: ((1, 0),), 14: ((0, 3),)}
DRAPE = 2.0  # m over the terrain the curves for ParaView are drawn
STEP = 5.0  # m between their points, so the drape follows the ground


def road_line(path):
    """The SS36's centre line, (x, y) in the grid's frame.

    The kmz is a zipped kml of lon/lat in degrees (EPSG:4326), reprojected to
    UTM 32N here -- the one place this repository needs a projection, which is
    why rasterio comes in and only here."""
    from rasterio.warp import transform

    with zipfile.ZipFile(path) as archive:
        kml = ET.fromstring(archive.read("doc.kml"))
    coords = kml.findall(".//{http://www.opengis.net/kml/2.2}LineString/"
                         "{http://www.opengis.net/kml/2.2}coordinates")
    if len(coords) != 1:
        raise SystemExit(f"{path}: {len(coords)} line strings, expected one")
    lonlat = np.array([[float(v) for v in t.split(",")[:2]] for t in coords[0].text.split()])
    east, north = transform("EPSG:4326", "EPSG:32632", lonlat[:, 0], lonlat[:, 1])
    here = np.column_stack([east, north]) - planes.M[:2]
    print(f"  {Path(path).name}: {len(here)} points, {LineString(here).length:.0f} m, "
          f"({here[0, 0]:.0f},{here[0, 1]:.0f}) to ({here[-1, 0]:.0f},{here[-1, 1]:.0f})")
    return LineString(here)


def ribbon(ring, line):
    """The piece between the ring's road-facing arc and the road itself."""
    first, last = ring[ARC[0]], ring[ARC[1]]
    along = substring(line, *sorted([line.project(Point(first)), line.project(Point(last))]))
    print(f"  arc {ARC[0]}..{ARC[1]}: ({first[0]:.0f},{first[1]:.0f}) to "
          f"({last[0]:.0f},{last[1]:.0f}), {line.distance(Point(first)):.0f} and "
          f"{line.distance(Point(last)):.0f} m off the road, which it rides for "
          f"{along.length:.0f} m")
    walk = np.array(along.coords)
    if np.linalg.norm(walk[0] - first) > np.linalg.norm(walk[-1] - first):
        walk = walk[::-1]
    return Polygon(np.vstack([[first], walk, [last]])).buffer(0)


def extended(path, line):
    """The surveyed deposit carried onto the road along that arc alone."""
    given = np.loadtxt(path, skiprows=1, usecols=(0, 1)) - planes.M[:2]
    clean = Polygon(given).buffer(0)
    ring = np.array(clean.exterior.coords)[:-1]
    spikes = [i for i, p in enumerate(given) if np.hypot(*(ring - p).T).min() > 1e-6]
    print(f"  {Path(path).name}: {len(given)} vertices, "
          f"{'valid' if Polygon(given).is_valid else 'self-crossing'} -- "
          f"{len(spikes)} spike(s) {spikes} dropped, {len(ring)} left")

    whole = clean.union(ribbon(ring, line))
    grown = np.array(whole.exterior.coords)[:-1]
    gone = [i for i, p in enumerate(ring) if np.hypot(*(grown - p).T).min() > 1e-6]
    print(f"  {len(ring) - len(gone)} of {len(ring)} vertices kept verbatim, "
          f"{len(gone)} swallowed: {gone[0]}..{gone[-1]}")
    if gone != list(range(gone[0], gone[-1] + 1)):
        raise SystemExit(f"swallowed {gone}, which is not one run: the ribbon eats elsewhere")
    if gone[0] <= ARC[0] or gone[-1] >= ARC[1]:
        raise SystemExit(f"swallowed {gone[0]}..{gone[-1]}, outside the arc {ARC}")
    print(f"  the deposit goes from {clean.area:,.0f} to {whole.area:,.0f} m2 "
          f"(+{100 * whole.area / clean.area - 100:.0f}%), {len(grown)} vertices")
    return grown


def zero_level(X, Y, F):
    """The curve F = 0, as line strings: marching squares over the samples,
    each crossing read off its own edge by linear interpolation, and the
    segments joined by shapely. The same curve a contour plot draws, with no
    plotting library behind it."""
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    segments = []
    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            f = [F[i + dx, j + dy] for dx, dy in corners]
            index = sum(1 << k for k, v in enumerate(f) if v < 0)
            if index not in JOINS:
                continue
            on = {}
            for edge in range(4):
                a, b = edge, (edge + 1) % 4
                if (f[a] < 0) != (f[b] < 0):
                    t = f[a] / (f[a] - f[b])
                    ax, ay = X[i + corners[a][0], j + corners[a][1]], Y[i + corners[a][0], j + corners[a][1]]
                    bx, by = X[i + corners[b][0], j + corners[b][1]], Y[i + corners[b][0], j + corners[b][1]]
                    on[edge] = (ax + t * (bx - ax), ay + t * (by - ay))
            segments += [(on[a], on[b]) for a, b in JOINS[index]]
    merged = linemerge(segments)
    return list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]


def wedge(con, terrain, sink, step=2.0):
    """The ground inside the conglomerate that still lies under the upper
    plane: the conglomerate cut along that plane's outcrop, north piece."""
    x0, y0, x1, y1 = con.bounds
    X, Y = np.meshgrid(np.arange(x0 - step, x1 + step, step),
                       np.arange(y0 - step, y1 + step, step), indexing="ij")
    xy = np.column_stack([X.ravel(), Y.ravel()])
    over = (terrain(xy) - planes.surveyed(xy, sink + planes.DROPS[0])).reshape(X.shape)
    branches = [np.array(line.coords) for line in zero_level(X, Y, over)]
    outcrop = max(branches, key=len)
    print(f"  the upper plane outcrops there in {len(branches)} branch(es), the longest of "
          f"{len(outcrop)} points, x [{outcrop[:, 0].min():.0f},{outcrop[:, 0].max():.0f}] "
          f"y [{outcrop[:, 1].min():.0f},{outcrop[:, 1].max():.0f}]")
    pieces = list(split(con, LineString(outcrop).simplify(1.0)).geoms)
    north = [p for p in pieces if p.centroid.y > con.centroid.y]
    if len(pieces) != 2 or len(north) != 1:
        raise SystemExit(f"the outcrop cuts the conglomerate into {len(pieces)} pieces")
    print(f"  the wedge it cuts off, which the region takes in: {north[0].area:,.0f} m2, "
          f"x [{north[0].bounds[0]:.0f},{north[0].bounds[2]:.0f}] "
          f"y [{north[0].bounds[1]:.0f},{north[0].bounds[3]:.0f}]")
    return north[0]


def behind(con, terrain, sink):
    """The ground behind the conglomerate, out to the end of y."""
    ring = np.array(con.exterior.coords)[:-1]
    west, east = int(ring[:, 0].argmin()), int(ring[:, 0].argmax())
    one = np.roll(ring, -west, axis=0)[: (east - west) % len(ring) + 1]
    other = np.roll(ring, -east, axis=0)[: (west - east) % len(ring) + 1][::-1]
    chain = one if one[:, 1].mean() > other[:, 1].mean() else other
    print(f"  conglomerate: west extreme ({ring[west, 0]:.1f},{ring[west, 1]:.1f}), east "
          f"({ring[east, 0]:.1f},{ring[east, 1]:.1f}); the closure takes its own "
          f"{len(chain)} vertices between them, topping out at y {chain[:, 1].max():.1f}")
    drawn = EAST[EAST[:, 1] < YMAX]
    gap = np.linalg.norm(drawn[0] - chain[-1])
    print(f"  the drawn line starts {gap:.1f} m from that east extreme, which it is pinned "
          f"to; {len(drawn)} of its points stand below y {YMAX:g}")
    if gap > 10.0:
        raise SystemExit("the drawn line does not meet the conglomerate's east extreme")
    walk = np.vstack([chain, drawn[1:], [[drawn[-1, 0], YMAX], [chain[0, 0], YMAX]]])

    whole = unary_union([Polygon(walk), wedge(con, terrain, sink)])
    ring = np.array(whole.exterior.coords)[:-1]
    print(f"  behind it: {whole.area:,.0f} m2, x [{ring[:, 0].min():.1f},"
          f"{ring[:, 0].max():.1f}] y [{ring[:, 1].min():.1f},{ring[:, 1].max():.1f}], "
          f"{len(ring)} vertices, {'simple' if whole.is_valid else 'SELF-CROSSING'}")
    return ring


def densified(walk, closed):
    """A point every STEP, so a draped curve follows the ground."""
    pairs = zip(walk, np.roll(walk, -1, axis=0)) if closed else zip(walk[:-1], walk[1:])
    fine = [walk[0]]
    for a, b in pairs:
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / STEP)))
        fine += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(fine[:-1] if closed else fine)


def write_region(path, ring, terrain):
    """The outline as the case reads it: X Y Z in UTM32N, z off the terrain."""
    np.savetxt(path, np.column_stack([ring + planes.M[:2], terrain(ring) + planes.M[2]]),
               fmt="%.6f", header="//X Y Z", comments="")
    print(f"wrote {path}: {len(ring)} vertices")


def write_curve(path, walk, terrain, closed):
    """The same line for ParaView, in the grid's frame and over the ground."""
    fine = densified(walk, closed)
    ends = np.arange(len(fine))
    lines = np.column_stack([ends, np.roll(ends, -1)])
    meshio.write_points_cells(
        path, np.column_stack([fine, terrain(fine) + DRAPE]),
        [("line", lines if closed else lines[:-1])],
    )
    print(f"wrote {path}: {len(fine)} points")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)
    if case.deposit is None or case.back_deposit is None or case.conglomerate is None:
        raise SystemExit(f"{case.name}: this is Rialba's own geology -- it wants a "
                         "conglomerate, a deposit and a back_deposit to write")

    line = road_line(root / ROAD)
    grown = extended(root / SURVEYED, line)

    terrain = load_dtm_interpolator(case.points)
    con = planes.outline_of(case.conglomerate, "conglomerate")
    ring = behind(con, terrain, planes.under_the_slope(case))

    write_region(case.deposit, grown, terrain)
    write_region(case.back_deposit, ring, terrain)
    case.output.mkdir(parents=True, exist_ok=True)
    write_curve(case.output / f"{case.deposit.stem}.vtu", grown, terrain, True)
    write_curve(case.output / f"{case.back_deposit.stem}.vtu", ring, terrain, True)
    write_curve(case.output / "ss36.vtu", np.array(line.coords), terrain, False)


if __name__ == "__main__":
    main()
