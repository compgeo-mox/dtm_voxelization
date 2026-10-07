"""The mesh's profile on the plane that stands normal to the case's fractures,
flattened onto xy: `python -m dtm_voxelization.section [CASE.toml]`, or the
file itself from an editor, which takes CASE below.

The plane is VERTICAL and passes through the middle of every fracture: its
normal is horizontal, and is the direction least occupied by the fractures'
own centres in plan -- the smaller eigenvector of their covariance -- so the
plane is the one that best threads them. Each fracture counts once, by its
area-weighted centre, or the largest surface would drag the plane onto itself.

Standing normal to the fractures and threading their centres are two different
demands, and on a chain of fractures they pull apart. At Rialba, threading
them passes 0.1 to 2.2 m from each centre and meets them at 63 to 89 degrees;
the most perpendicular vertical plane instead averages 83 degrees but misses
the centres by up to 30 m. The profile is wanted through the fractures, so the
first wins.

A cavity is not a fracture and takes no part in any of this: surfaces whose
name contains one of NOT_FRACTURES are left out, of the plane and of the
section both.

Everything is then written in the plane's own two axes -- x along it,
horizontal, y up -- as a flat drawing at z = 0:

- profile.txt / .vtp: the mountain's outline, the boundary of the section as a
  closed polyline wound counter-clockwise, smoothed off the grid's staircase.
  It is the whole boundary, the domain's own box included, since that is what
  closes it -- 1.8 km of section, far more than a 2D model wants.
- domain.txt / .vtp: that outline cut back to BOX, the block around the
  fractures. This is the one to hand to a mesher as a list of vertices; the
  fractures below are cut back to it too.
- fracture_<name>.txt / .vtp: one per fracture, points running from the top
  down, cut back to what falls inside the domain and carried up to meet it, so
  every fracture starts ON the boundary and not a stair short of it.

The .txt files are two columns, x and y, which is what a 2D model wants; the
.vtp are the same points as lines, to look at. section.vtp, section_surfaces
.vtp and section_plane.vtp keep the full 3D cut, in place.
"""

import sys
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import LineString, Polygon
from shapely.geometry import box as rectangle
from shapely.geometry.polygon import orient
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPlane, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkCutter
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter, vtkXMLUnstructuredGridReader

if __package__:
    from .case import load_case
    from .clean_stl import to_polydata
    from .fracture_surface import inside
    from .mesh import require
    from .shift_fracture import M  # the same Rialba mean
    from .surfaces import in_grid_frame
    from .terrain import grid_triangles
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.case import load_case
    from dtm_voxelization.clean_stl import to_polydata
    from dtm_voxelization.fracture_surface import inside
    from dtm_voxelization.mesh import require
    from dtm_voxelization.shift_fracture import M
    from dtm_voxelization.surfaces import in_grid_frame
    from dtm_voxelization.terrain import grid_triangles

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button
NOT_FRACTURES = ("cavita",)  # a cavity is a hole in the rock, not a fracture
BOX = ((-300.0, -200.0), (150.0, 100.0))  # the 2D model's own window on the section
SMOOTH_PASSES = 10  # Taubin pairs over the outline, enough to take the stairs off
WEIGHTS = (0.5, -0.53)  # Taubin's: a step towards the neighbours, a wider one back
# a surveyed plane, as 0.2230 X + 0.4732 Y - 0.8523 Z + d = 0 with d / -0.8523 given,
# and a second one DROPS below it
PLANE = (0.2230, 0.4732, -0.8523, 3.2214e06)
PLANE_CRS = 3003  # Monte Mario / Italy zone 1, the Gauss-Boaga the plane was fitted in
DROPS = (0.0, 20.0)
SHEET = 20  # vertices a side of the sheet each plane is drawn as


def fractures(case):
    """The case's surfaces that are fractures, cavities left out."""
    kept = [s for s in case.surfaces if not any(n in s.name for n in NOT_FRACTURES)]
    dropped = [s.name for s in case.surfaces if s not in kept]
    if dropped:
        print(f"  leaving out {', '.join(dropped)}: not fractures")
    if not kept:
        raise SystemExit(f"{case.name}: no fractures left to stand normal to")
    return kept


def through_the_fractures(case, surfaces):
    """(normal, origin) of the vertical plane threading the fractures' middles."""
    centres, normals = [], []
    for surface in surfaces:
        tri = in_grid_frame(case, surface)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cross, axis=1) / 2
        centres.append((tri.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum())
        # the area-weighted mean normal describes the fracture's own attitude
        mean = (cross / np.linalg.norm(cross, axis=1, keepdims=True) * area[:, None]).sum(axis=0)
        normals.append(mean / np.linalg.norm(mean))
    centres, normals = np.array(centres), np.array(normals)

    origin = centres.mean(axis=0)
    spread, directions = np.linalg.eigh(np.cov((centres[:, :2] - origin[:2]).T, bias=True))
    normal = np.array([*directions[:, 0], 0.0])  # horizontal: the plane is vertical
    print(
        f"  the fractures' centres spread {np.sqrt(spread[1]):.0f} m along the plane and "
        f"{np.sqrt(spread[0]):.1f} m across it"
    )
    print(f"  plane: normal {np.round(normal, 4)} through {np.round(origin, 1)}")
    for surface, centre, mean in zip(surfaces, centres, normals):
        print(
            f"    {surface.name}: centre {abs((centre - origin) @ normal):.1f} m off the plane, "
            f"met at {90 - np.degrees(np.arcsin(abs(mean @ normal))):.0f} degrees"
        )
    return normal, origin


def flatten(points, normal, origin):
    """The plane's own coordinates: x along it and horizontal, y up."""
    along = np.cross(normal, [0.0, 0.0, 1.0])
    return (points - origin) @ np.column_stack([along / np.linalg.norm(along), [0, 0, 1.0]])


def loops_of(edges):
    """Chains of the given edges, as lists of point indices. A chain closes
    when it comes back to where it started."""
    ends = {}
    for a, b in edges:
        ends.setdefault(a, []).append(b)
        ends.setdefault(b, []).append(a)
    seen, chains = set(), []
    for start in ends:
        if start in seen:
            continue
        chain, node = [start], start
        seen.add(start)
        while True:
            step = [n for n in ends[node] if n not in seen]
            if not step:
                break
            node = step[0]
            seen.add(node)
            chain.append(node)
        chains.append(chain)
    return chains


def outline(points, polygons):
    """The section's boundary: the edges only one of its polygons owns, chained
    and wound counter-clockwise. The longest chain is the outline itself; the
    others are the holes the cracks and the carving leave."""
    edges = np.vstack([np.column_stack([p, np.roll(p, -1)]) for p in polygons])
    _, first, counts = np.unique(np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True)
    rim = edges[first[counts == 1]]
    chains = sorted(loops_of(rim), key=len, reverse=True)
    print(
        f"  the section's boundary: {len(rim):,} edges in {len(chains)} chains, the longest "
        f"{len(chains[0]):,} points; the rest are holes"
    )
    loop = np.array(chains[0])
    x, y = points[loop, 0], points[loop, 1]
    if np.sum((np.roll(x, -1) - x) * (np.roll(y, -1) + y)) > 0:  # clockwise
        loop = loop[::-1]
    return loop


def from_the_top(points, segments):
    """The cut of one surface as ONE line, running downwards.

    The cut comes out in pieces for two reasons, both small: it branches, where
    three segments meet at a point and a walk has to stop, and it breaks, where
    the plane grazes out of the surface and back in near its edge. The pieces
    are sewn from the highest downwards, each time jumping to whichever loose
    end is nearest, so what the file holds is a single continuous line."""
    chains = [list(chain) for chain in loops_of(segments)]
    for n, chain in enumerate(chains):
        if points[chain[0], 1] < points[chain[-1], 1]:
            chains[n] = chain[::-1]

    line = chains.pop(max(range(len(chains)), key=lambda n: points[chains[n][0], 1]))
    jumps = []
    while chains:
        tail = points[line[-1]]
        reach = [
            min(np.linalg.norm(points[chain[0]] - tail), np.linalg.norm(points[chain[-1]] - tail))
            for chain in chains
        ]
        chain = chains.pop(int(np.argmin(reach)))
        if np.linalg.norm(points[chain[-1]] - tail) < np.linalg.norm(points[chain[0]] - tail):
            chain = chain[::-1]
        jumps.append(min(reach))
        line += chain[1:] if np.allclose(points[chain[0]], tail) else chain
    if jumps:
        print(
            f"    sewn from {len(jumps) + 1} pieces, jumping at most {max(jumps):.2f} m "
            f"({sum(j > 0 for j in jumps)} of the joins were a real gap)"
        )
    return [np.array(line)]


def smooth(loop):
    """The outline with the grid's staircase taken off it.

    The cut of a hex grid is a staircase: at Rialba 824 of its 832 segments run
    along x or y and 351 corners are square, each stair half a cell high, and
    none of that is terrain. Taubin's pair of passes -- a step towards the
    neighbours, a wider one back -- leaves the line on the middle of the stairs
    without the shrinking a plain Laplacian brings, which here would cut the
    ridges off. Points on the domain's own sides and floor are held, so the box
    stays a box and only the carved top is touched.
    """
    keep = np.append(np.linalg.norm(np.diff(loop, axis=0), axis=1) > 1e-6, True)
    if not keep.all():
        print(f"  dropping {(~keep).sum()} outline points the cut left on top of each other")
    loop = loop[keep]

    lo, hi = loop.min(axis=0), loop.max(axis=0)
    pinned = (
        np.isclose(loop[:, 0], lo[0])
        | np.isclose(loop[:, 0], hi[0])
        | np.isclose(loop[:, 1], lo[1])
    )
    out = loop.copy()
    for _ in range(SMOOTH_PASSES):
        for weight in WEIGHTS:
            middle = (np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0)) / 2
            step = weight * (middle - out)
            step[pinned] = 0.0
            out += step
    moved = np.linalg.norm(out - loop, axis=1)
    print(
        f"  smoothed the outline: {SMOOTH_PASSES} Taubin passes over "
        f"{len(loop) - pinned.sum():,} free points ({pinned.sum()} held on the domain's "
        f"own sides and floor), moving them {moved.mean():.2f} m on average and "
        f"{moved.max():.2f} m at most"
    )
    return out


def clip(loop, corners):
    """The outline cut back to the model's own box, counter-clockwise.

    The section runs over the whole domain and the 2D model wants the block
    around the fractures, so the polygon is intersected with BOX. A staircase
    that has been smoothed can touch itself, which shapely calls invalid and
    will not intersect, so it is repaired first; if the box happens to catch
    more than one piece of rock the largest is the model's."""
    (x0, y0), (x1, y1) = corners
    polygon = Polygon(loop)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        print("  the outline touches itself somewhere; repaired before clipping")
    cut = polygon.intersection(rectangle(x0, y0, x1, y1))
    if cut.is_empty:
        raise SystemExit(f"the box {corners} meets nothing of the section")
    if cut.geom_type == "MultiPolygon":
        pieces = sorted(cut.geoms, key=lambda piece: piece.area, reverse=True)
        print(
            f"  the box catches {len(pieces)} separate pieces of rock; keeping the largest, "
            f"{pieces[0].area:,.0f} of {cut.area:,.0f} m2"
        )
        cut = pieces[0]
    if cut.interiors:
        print(f"  leaving out {len(cut.interiors)} hole(s) the carving opened inside the box")
    out = np.array(orient(Polygon(cut.exterior)).exterior.coords)[:-1]
    print(
        f"  clipped to x {x0:g} .. {x1:g}, y {y0:g} .. {y1:g}: {len(out):,} points, "
        f"{cut.area:,.0f} m2"
    )
    return out


def onto_the_outline(polygon, line):
    """The fracture's line carried up to END on the outline.

    Cutting the trace back to what lies inside the polygon leaves it a stair
    short of the top, which for a 2D model is a fracture that stops in the air.
    The first segment is produced upwards instead, and the line starts where it
    crosses the outline -- on the border, not past it."""
    start, step = line[0], line[0] - line[1]
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edge = b - a
    denominator = step[0] * edge[:, 1] - step[1] * edge[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = a - start
        t = (gap[:, 0] * edge[:, 1] - gap[:, 1] * edge[:, 0]) / denominator
        s = (gap[:, 0] * step[1] - gap[:, 1] * step[0]) / denominator
    hit = (denominator != 0) & (t > 0) & (s >= 0) & (s <= 1)
    if not hit.any():
        raise ValueError("the fracture's top segment, produced upwards, misses the outline")
    rise = t[hit].min()
    print(f"    carried up {rise * np.linalg.norm(step):.2f} m to end on the outline")
    return np.vstack([start + rise * step, line])


def surveyed(points, drop):
    """The surveyed plane's z, DROP metres lower, over the given x and y.

    The plane was fitted in Monte Mario / Italy zone 1 (Gauss-Boaga), whose
    easting runs a million metres ahead of UTM's, so x and y go back to UTM --
    the grid's frame is UTM minus the cloud's mean M -- and on to Gauss-Boaga
    before the equation is used. Taken in UTM instead it puts the plane 262 km
    underground, which is that million metres times 0.2230 / 0.8523.

    Its Z is an elevation above the sea, so the cloud's mean comes off it as
    it does off everything else. That is how the whole Gauss-Boaga lineage of
    this case is written -- Rialba_DTM2x2.txt, DTMRialba5m_4Cubit, the Cubit
    pipeline's own local frame -- where x and y are shifted and z never is."""
    a, b, c, offset = PLANE
    east, north = transform(
        CRS.from_epsg(32632),
        CRS.from_epsg(PLANE_CRS),
        points[:, 0] + M[0],
        points[:, 1] + M[1],
    )
    return -(a / c * np.array(east) + b / c * np.array(north) + offset) - M[2] - drop


def write_sheet(path, drop, bounds):
    """One surveyed plane as a sheet over the mesh, to look at in ParaView.

    It is drawn as a grid rather than a quad because the two map projections
    are not quite parallel, so the plane is not quite flat in the grid's
    frame."""
    box = np.array(bounds).reshape(3, 2)
    x, y = np.meshgrid(
        np.linspace(*box[0], SHEET), np.linspace(*box[1], SHEET), indexing="ij"
    )
    points = np.column_stack([x.ravel(), y.ravel()])
    sheet = np.column_stack([points, surveyed(points, drop)])
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(to_polydata(sheet, grid_triangles(SHEET, SHEET)))
    if not writer.Write():
        raise RuntimeError(f"vtkXMLPolyDataWriter failed on {path}")
    print(
        f"wrote {path.name}: the plane {drop:g} m down, z from {sheet[:, 2].min():.0f} to "
        f"{sheet[:, 2].max():.0f} over the mesh"
    )


def on_the_section(drop, normal, origin, polygon):
    """One surveyed plane's trace on the vertical plane, cut to the domain.

    The two planes cross the vertical one along a straight line, so the trace
    is read at the domain's own ends and then cut by the polygon, which leaves
    it starting and ending ON the boundary, as the fractures do. It comes back
    empty if the plane passes over the rock altogether."""
    along = np.cross(normal, [0.0, 0.0, 1.0])
    along /= np.linalg.norm(along)
    reach = np.array([polygon[:, 0].min(), polygon[:, 0].max()])
    at = origin[:2] + reach[:, None] * along[:2]
    line = np.column_stack([reach, surveyed(at, drop) - origin[2]])

    cut = LineString(line).intersection(Polygon(polygon))
    if cut.is_empty:
        print(
            f"    the plane {drop:g} m down misses the domain: over it the trace runs from "
            f"y = {line[0, 1]:.0f} to {line[1, 1]:.0f}, the rock from "
            f"{polygon[:, 1].min():.0f} to {polygon[:, 1].max():.0f}"
        )
        return []
    pieces = list(cut.geoms) if cut.geom_type == "MultiLineString" else [cut]
    chains = []
    for piece in pieces:
        chain = np.array(piece.coords)
        chains.append(chain[::-1] if chain[0, 1] < chain[-1, 1] else chain)
    chains.sort(key=lambda chain: -chain[0, 1])
    print(
        f"    the plane {drop:g} m down crosses the domain in {len(chains)} piece(s), "
        f"y from {chains[0][0, 1]:.1f} down to {chains[-1][-1, 1]:.1f}"
    )
    return chains


def write_flat(path, chains):
    """Two columns of x and y for the model, and the same as lines to look at."""
    rows = np.vstack(chains)
    np.savetxt(path.with_suffix(".txt"), rows, fmt="%.4f", header="x y", comments="")

    flat = np.column_stack([rows, np.zeros(len(rows))])
    lines, at = [], 0
    for chain in chains:
        lines += [[at + k, at + k + 1] for k in range(len(chain) - 1)]
        at += len(chain)
    mesh = vtkPolyData()
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(flat), deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 2 * len(lines) + 1, 2, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.array(lines, dtype=np.int64).ravel(), deep=True),
    )
    mesh.SetPoints(vtk_points)
    mesh.SetLines(cells)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path.with_suffix(".vtp")))
    writer.SetInputData(mesh)
    if not writer.Write():
        raise RuntimeError(f"vtkXMLPolyDataWriter failed on {path}")
    print(
        f"wrote {path.with_suffix('.txt').name} and {path.with_suffix('.vtp').name}: "
        f"{len(rows):,} points in {len(chains)} chain(s)"
    )


def write_plane(path, normal, origin, bounds):
    """The plane itself, as the rectangle of it that covers the mesh."""
    along = np.cross(normal, [0.0, 0.0, 1.0])
    along /= np.linalg.norm(along)
    up = np.array([0.0, 0.0, 1.0])
    box = np.array(bounds).reshape(3, 2)
    corners = np.array(np.meshgrid(*box)).reshape(3, -1).T - origin
    reach = [(corners @ along).min(), (corners @ along).max()]
    rise = [(corners @ up).min(), (corners @ up).max()]
    quad = np.array(
        [
            origin + along * reach[a] + up * rise[b]
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
    )
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(to_polydata(quad, np.array([[0, 1, 2], [0, 2, 3]])))
    if not writer.Write():
        raise RuntimeError(f"vtkXMLPolyDataWriter failed on {path}")
    print(
        f"wrote {path}: the plane over {reach[1] - reach[0]:.0f} x {rise[1] - rise[0]:.0f} m "
        f"of the mesh"
    )


def cut(data, plane, path=None):
    """The section of one dataset by the plane, written if asked for.

    The cut is cleaned before anything else looks at it: a surface read from an
    STL carries each triangle's own three corners, so its section comes out as
    loose segments that share no point and chain into nothing."""
    cutter = vtkCutter()
    cutter.SetInputData(data)
    cutter.SetCutFunction(plane)
    clean = vtkCleanPolyData()
    clean.SetInputConnection(cutter.GetOutputPort())
    clean.Update()
    out = clean.GetOutput()
    if path is not None:
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(out)
        writer.SetDataModeToAppended()
        writer.SetCompressorTypeToLZ4()
        if not writer.Write():
            raise RuntimeError(f"vtkXMLPolyDataWriter failed on {path}")
        print(f"wrote {path}: {out.GetNumberOfCells():,} cells, {out.GetNumberOfPoints():,} points")
    return out


def cells_of(polydata, which):
    """A cut's cells as lists of point indices: its polygons, or its lines."""
    array = polydata.GetPolys() if which == "polys" else polydata.GetLines()
    if not array.GetNumberOfCells():
        return []
    offsets = vtk_to_numpy(array.GetOffsetsArray())
    links = vtk_to_numpy(array.GetConnectivityArray())
    return [links[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)
    if not case.surfaces:
        raise SystemExit(f"{case.name}: no surfaces, so no plane to stand normal to")

    surfaces = fractures(case)
    normal, origin = through_the_fractures(case, surfaces)
    plane = vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    mesh = case.detached_path if case.detached_path.exists() else case.grid_path
    require(mesh, "grid")
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(mesh))
    reader.Update()
    print(f"  cutting {mesh}: {reader.GetOutput().GetNumberOfCells():,} cells")
    section = cut(reader.GetOutput(), plane, case.output / "section.vtp")
    write_plane(case.output / "section_plane.vtp", normal, origin, reader.GetOutput().GetBounds())

    flat = flatten(vtk_to_numpy(section.GetPoints().GetData()), normal, origin)
    rim = outline(flat, cells_of(section, "polys"))
    whole = smooth(flat[rim])
    write_flat(case.output / "profile", [np.vstack([whole, whole[0]])])
    polygon = clip(whole, BOX)
    write_flat(case.output / "domain", [np.vstack([polygon, polygon[0]])])

    for surface in surfaces:
        tri = in_grid_frame(case, surface)
        faces = np.arange(3 * len(tri)).reshape(-1, 3)
        trace = cut(to_polydata(tri.reshape(-1, 3), faces), plane)
        here = flatten(vtk_to_numpy(trace.GetPoints().GetData()), normal, origin)
        chains = from_the_top(here, cells_of(trace, "lines"))
        within = inside(polygon, here)
        kept = [chain[within[chain]] for chain in chains]
        kept = [here[chain] for chain in kept if len(chain) > 1]
        print(
            f"  {surface.name}: {sum(len(c) for c in chains):,} points on the plane, "
            f"{sum(len(c) for c in kept):,} inside the outline"
        )
        kept[0] = onto_the_outline(polygon, kept[0])
        write_flat(case.output / f"fracture_{surface.name}", kept)

    for drop in DROPS:
        write_sheet(case.output / f"plane_{drop:g}m.vtp", drop, reader.GetOutput().GetBounds())
        chains = on_the_section(drop, normal, origin, polygon)
        if chains:
            write_flat(case.output / f"plane_{drop:g}m", chains)


if __name__ == "__main__":
    main()
