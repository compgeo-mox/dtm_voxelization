"""Export flat polygonal 'planes' as STL surfaces.

Pure geometry: each entry of `PLANES` is one polygon given by its corner
points, listed in order around the boundary, and nothing else is read --
no DTM, no point cloud. Corners may be

- `(x, y)`      -- the polygon is horizontal at the elevation `"z"`
                   (per plane, default `Z` below), or
- `(x, y, z)`   -- used as given, so the polygon can sit at any tilt.

Adding a plane later means appending one more dict to `PLANES`;
`{"name": ..., "vertices": [...]}` is the whole contract. Any number of
corners (>= 3) is fine.

Each plane is written to `output/planes/<name>.stl`.
"""

from pathlib import Path

import numpy as np
import meshio

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output" / "planes"

Z = 0.0  # default elevation for planes whose corners are 2D

PLANES = [
    {
        "name": "piano_1",
        "vertices": [
            (664.860, 446.232),
            (669.367, 446.232),
            (669.367, 446.882),
            (664.860, 446.882),
        ],
    },
    {
        "name": "piano_2",
        "vertices": [
            (682.285, 444.996),
            (686.346, 444.996),
            (686.346, 446.238),
            (682.850, 446.238),
        ],
    },
]


def build_plane(plane):
    """One PLANES entry -> (points (N, 3), triangles (M, 3)).

    Triangulated as a fan from the first corner -- exact for convex
    polygons (and for any polygon whose first corner sees the whole
    interior)."""
    vertices = np.asarray(plane["vertices"], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError(
            f"{plane['name']}: vertices must be (N, 2) or (N, 3), "
            f"got {vertices.shape}"
        )
    if len(vertices) < 3:
        raise ValueError(f"{plane['name']}: a polygon needs at least 3 corners")

    if vertices.shape[1] == 2:
        z = plane.get("z", Z)
        vertices = np.column_stack([vertices, np.full(len(vertices), z)])

    triangles = np.array(
        [(0, k, k + 1) for k in range(1, len(vertices) - 1)], dtype=int
    )
    return vertices, triangles


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for plane in PLANES:
        points, triangles = build_plane(plane)
        out_path = OUTPUT_DIR / f"{plane['name']}.stl"
        meshio.write_points_cells(out_path, points, [("triangle", triangles)])
        print(
            f"wrote {out_path} -- {len(points)} corners, "
            f"{len(triangles)} triangles, z = {points[:, 2].min():g}"
            + (
                ""
                if points[:, 2].min() == points[:, 2].max()
                else f" .. {points[:, 2].max():g}"
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
