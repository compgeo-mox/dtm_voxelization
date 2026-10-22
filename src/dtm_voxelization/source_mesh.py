"""A plain hex mesh of the box around the source points, for SPEED:
`python -m dtm_voxelization.source_mesh [CASE.toml]`, or the file itself from
an editor, which takes CASE below.

It is the same box the grid refines around the source points (grid.py's
`source_box`: SOURCE_BACK of their middle downhill, SOURCE_REACH towards the
fractures, the terrain over it plus the case's region padding), filled with the
plainest mesh there is -- cells of about CELL metres, all of one size, no
transition templates anywhere. In plan it is therefore a rectangle, and the
only thing that breaks the brick is the terrain, which carves the top exactly
as it carves the big grid: a hex stays when its top face is under the terrain
over its own centre.

Boundary quads carry the labels speed.py gives the big mesh, so a .mate file
written for one works on the other:

- LATERAL_BOTTOM_TAG on the four sides and the floor, which are plain
  rectangles since nothing is refined;
- TOP_TAG on everything else, which here is the carved top and the vertical
  steps it leaves.

This mesh has nothing to do with grid.vtu: it is built from the terrain
straight, so it needs no step of the pipeline run first, and it is not cut by
any fracture. Writes source_mesh.vtu next to the case's output, and
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
    from .mesh import build_face_table
    from .speed import LATERAL_BOTTOM_TAG, TOP_TAG, count_inverted_hexes, write_speed_mesh
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization import frame
    from dtm_voxelization import terrain as terrain_surfaces
    from dtm_voxelization.case import load_case
    from dtm_voxelization.grid import around_boxes, source_box
    from dtm_voxelization.mesh import build_face_table
    from dtm_voxelization.speed import (
        LATERAL_BOTTOM_TAG,
        TOP_TAG,
        count_inverted_hexes,
        write_speed_mesh,
    )

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button
CELL = 2.0  # m wanted of a cell's side; each axis takes the whole count nearest to it


def brick(box, cell):
    """(points, hexes) of the plain grid filling the box, cells of about `cell`.

    Each axis is divided into the whole number of cells nearest to its own
    length over `cell`, so the box is kept exactly and the three sides of a
    cell come out within a few per cent of each other rather than exactly
    equal. Hexes are numbered in VTK order, bottom face first."""
    lo = np.array([box[0], box[2], box[4]])
    hi = np.array([box[1], box[3], box[5]])
    counts = np.maximum(1, np.round((hi - lo) / cell).astype(int))
    nx, ny, nz = counts
    axes = [np.linspace(lo[k], hi[k], counts[k] + 1) for k in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # node (i, j, k) is at i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = (i * (ny + 1) + j) * (nz + 1) + k
    step_i, step_j = (ny + 1) * (nz + 1), nz + 1
    hexes = np.stack(
        [
            base,
            base + step_i,
            base + step_i + step_j,
            base + step_j,
            base + 1,
            base + step_i + 1,
            base + step_i + step_j + 1,
            base + step_j + 1,
        ],
        axis=-1,
    ).reshape(-1, 8)
    print(
        f"  brick {nx} x {ny} x {nz} = {len(hexes):,} hexes, cells of "
        f"{(hi[0] - lo[0]) / nx:.3f} x {(hi[1] - lo[1]) / ny:.3f} x "
        f"{(hi[2] - lo[2]) / nz:.3f} m",
        flush=True,
    )
    return points, hexes


def compact(points, hexes):
    """The mesh with the points no hex uses dropped."""
    used, renumbered = np.unique(hexes, return_inverse=True)
    return points[used], renumbered.reshape(-1, 8)


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
        raise SystemExit(f"{case.name}: no [grid] source_points, so no box to mesh")

    terrain, rotation, center = terrain_surfaces.build(case)
    box = source_box(case, terrain, around_boxes(case, rotation, center))
    points, hexes = brick(box, CELL)

    keep = terrain.keep(points[hexes])
    print(f"  carved: {int(keep.sum()):,} of {len(hexes):,} hexes under the terrain")
    points, hexes = compact(points, hexes[keep])
    inverted = count_inverted_hexes(points, hexes)
    if inverted:
        raise ValueError(f"{inverted} inverted hexes -- SPEED needs positive Jacobians")

    quads, tags = tag_boundary(points, hexes)
    points = frame.to_dtm(points, rotation, center)

    case.output.mkdir(parents=True, exist_ok=True)
    mesh_vtu = case.output / "source_mesh.vtu"
    meshio.write_points_cells(mesh_vtu, points, [("hexahedron", hexes)], binary=True)
    print(f"wrote {mesh_vtu}: {len(hexes):,} hexes, {len(points):,} points")

    case.speed_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = case.speed_dir / f"{case.name}_source.mesh"
    write_speed_mesh(mesh_path, points, quads, tags, hexes)
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
