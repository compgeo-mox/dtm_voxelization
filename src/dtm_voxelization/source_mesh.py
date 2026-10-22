"""The box around the source points, cut out of the grid as a mesh of its own:
`python -m dtm_voxelization.source_mesh [CASE.toml]`, or the file itself from
an editor, which takes CASE below.

It is a piece of grid.vtu, not a second mesh of the same ground: the hexes
whose centre falls in the box grid.py refined around the source points
(`source_box`). So it is the same discretization as the big grid, down to the
lattice the cells sit on, and it follows the case's target_cells without a
single number repeated here. The box sits inside the level-3 block with the
AROUND_SCALE margin to spare, so what comes out is all of one cell size and
all of one piece -- both checked, and the run stops if they are not.

The top is the carving the grid already has. In plan it is a full rectangle of
columns, and the floor is flat, at the bottom of the level-3 block.

The labels are speed.py's own, so a .mate written for the big mesh works here:
LATERAL_BOTTOM_TAG on the four sides and the floor, TOP_TAG on the carved top,
and the hexes labelled by the two surveyed planes, ROCK_TAG down to the lower
one and BELOW_TAG under it.

Needs the `grid` step run first. Writes source_mesh.vtu next to the case's
output, with the material tag on it to look at, and
speed/<case>_source.mesh with speed/<case>_source_boundary.vtu beside it.
"""

import sys
from pathlib import Path

import meshio
import numpy as np

if __package__:
    from . import frame
    from . import terrain as terrain_surfaces
    from .case import load_case
    from .grid import around_boxes, source_box
    from .mesh import build_face_table, read_hex_mesh, require
    from .planes import materials
    from .speed import LATERAL_BOTTOM_TAG, TOP_TAG, count_inverted_hexes, write_speed_mesh
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization import frame
    from dtm_voxelization import terrain as terrain_surfaces
    from dtm_voxelization.case import load_case
    from dtm_voxelization.grid import around_boxes, source_box
    from dtm_voxelization.mesh import build_face_table, read_hex_mesh, require
    from dtm_voxelization.planes import materials
    from dtm_voxelization.speed import (
        LATERAL_BOTTOM_TAG,
        TOP_TAG,
        count_inverted_hexes,
        write_speed_mesh,
    )

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button


def crop(points, hexes, box):
    """The hexes whose centre is in the box, renumbered onto their own points.

    Every one of them has to be the same size: the box lies inside the level-3
    block, so a coarser cell in there means the block did not cover the box and
    the piece would be a mesh of two resolutions."""
    middles = points[hexes].mean(axis=1)
    inside = np.ones(len(hexes), dtype=bool)
    for axis in range(3):
        inside &= (middles[:, axis] > box[2 * axis]) & (middles[:, axis] < box[2 * axis + 1])
    if not inside.any():
        raise SystemExit(f"no hex of the grid has its centre in the box {box}")

    kept = hexes[inside]
    sides = points[kept].max(axis=1) - points[kept].min(axis=1)
    finest = sides.min(axis=0)
    coarser = (sides > 1.01 * finest).any(axis=1)
    print(
        f"  {len(kept):,} of {len(hexes):,} hexes have their centre in the box, "
        f"cells of {finest[0]:.3f} x {finest[1]:.3f} x {finest[2]:.3f} m",
        flush=True,
    )
    if coarser.any():
        raise SystemExit(
            f"{int(coarser.sum()):,} of them are coarser than that, up to "
            f"{sides[coarser].max(axis=0).round(3)} m: the level-3 block does not cover "
            f"the box -- widen the inner region, or the refinement around the points"
        )

    used, renumbered = np.unique(kept, return_inverse=True)
    return points[used], renumbered.reshape(-1, 8)


def check_rectangle(points, hexes):
    """That the piece is a full rectangle of columns on a flat floor."""
    middles = points[hexes].mean(axis=1)
    side = (points[hexes].max(axis=1) - points[hexes].min(axis=1))[0]
    lo = points.min(axis=0)
    column = np.round((middles[:, :2] - lo[:2] - side[:2] / 2) / side[:2]).astype(int)
    columns = {tuple(c) for c in column}
    nx, ny = column.max(axis=0) + 1
    deep = np.bincount(column[:, 0] * ny + column[:, 1], minlength=nx * ny)
    floor = points[hexes].min(axis=1)[:, 2].min()
    print(
        f"  {nx} x {ny} columns of {deep.min()} to {deep.max()} cells, "
        f"{len(columns):,} of {nx * ny:,} standing: "
        f"{'a full rectangle' if len(columns) == nx * ny else 'HOLED'}, floor at {floor:.3f}",
        flush=True,
    )
    if len(columns) != nx * ny:
        raise SystemExit("the piece is not a full rectangle in plan")


def tag_boundary(points, hexes):
    """(quads, tags) of the boundary, the sides and floor apart from the top."""
    faces, owners = build_face_table(hexes)
    boundary = owners[:, 1] < 0
    quads, quad_owner = faces[boundary], owners[boundary, 0]

    q = points[quads]
    centers = q.mean(axis=1)
    normals = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    outward = np.einsum(
        "ij,ij->i", normals, centers - points[hexes[quad_owner]].mean(axis=1)
    )
    inward = int((outward <= 0).sum())
    if inward:
        raise ValueError(f"{inward} boundary quads point into the mesh")

    lo, hi = points.min(axis=0), points.max(axis=0)
    tol = 1e-6 * (hi - lo).max()
    side = (
        (np.abs(centers[:, 0] - lo[0]) < tol)
        | (np.abs(centers[:, 0] - hi[0]) < tol)
        | (np.abs(centers[:, 1] - lo[1]) < tol)
        | (np.abs(centers[:, 1] - hi[1]) < tol)
        | (np.abs(centers[:, 2] - lo[2]) < tol)
    )
    tags = np.where(side, LATERAL_BOTTOM_TAG, TOP_TAG)
    order = np.argsort(tags, kind="stable")
    print(
        f"  {len(faces):,} faces, {len(quads):,} on the boundary: "
        f"{int(side.sum()):,} tagged {LATERAL_BOTTOM_TAG} (sides and floor), "
        f"{int((~side).sum()):,} tagged {TOP_TAG} (the carved top)",
        flush=True,
    )
    return quads[order], tags[order]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)
    if case.source_points is None:
        raise SystemExit(f"{case.name}: no [grid] source_points, so no box to cut out")

    terrain, rotation, center = terrain_surfaces.build(case)
    box = source_box(case, terrain, around_boxes(case, rotation, center))

    require(case.grid_path, "grid")
    points, hexes = read_hex_mesh(case.grid_path)
    print(f"  {case.grid_path}: {len(hexes):,} hexes", flush=True)
    points, hexes = crop(points, hexes, box)
    check_rectangle(points, hexes)

    inverted = count_inverted_hexes(points, hexes)
    if inverted:
        raise ValueError(f"{inverted} inverted hexes -- SPEED needs positive Jacobians")
    quads, tags = tag_boundary(points, hexes)
    hex_tags = materials(case, points, hexes)  # in the grid's frame, as the planes are
    points = frame.to_dtm(points, rotation, center)

    case.output.mkdir(parents=True, exist_ok=True)
    mesh_vtu = case.output / "source_mesh.vtu"
    meshio.write_points_cells(
        mesh_vtu,
        points,
        [("hexahedron", hexes)],
        cell_data={"material": [hex_tags.astype(float)]},
        binary=True,
    )
    print(f"wrote {mesh_vtu}: {len(hexes):,} hexes, {len(points):,} points")

    case.speed_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = case.speed_dir / f"{case.name}_source.mesh"
    write_speed_mesh(mesh_path, points, quads, tags, hexes, hex_tags)
    print(
        f"wrote {mesh_path}: {len(points):,} nodes, {len(quads):,} quads + "
        f"{len(hexes):,} hexes"
    )
    used, renumbered = np.unique(quads, return_inverse=True)
    boundary_vtu = case.speed_dir / f"{case.name}_source_boundary.vtu"
    meshio.write_points_cells(
        boundary_vtu,
        points[used],
        [("quad", renumbered.reshape(-1, 4))],
        cell_data={"tag": [tags.astype(float)]},
    )
    print(f"wrote {boundary_vtu}")


if __name__ == "__main__":
    main()
