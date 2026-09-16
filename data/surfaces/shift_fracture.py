from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import Delaunay

# Mean of the original point cloud and the shift applied by the software
M = np.array([5.27837393e05, 5.08219147e06, 4.74976610e02])
S = np.array([-527000.0, -5082000.0, 0.0])


def read_points(path):
    # Lines are "id:x;y;z"
    rows = [line.split(":", 1)[1].split(";") for line in path.read_text().split()]
    return np.array(rows, dtype=float)


def triangulate(pts):
    # Delaunay in the local 2D frame of the best-fit plane
    centre = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - centre)
    print(f"  singular values {sv}, normal {vt[2]}")
    if sv[2] > 1e-8 * sv[0]:
        print(f"  WARNING: points are not coplanar (sv ratio {sv[2] / sv[0]:.3e})")
    uv = (pts - centre) @ vt[:2].T
    return Delaunay(uv).simplices


if __name__ == "__main__":
    folder = Path(__file__).parent
    for name in [
        "first_fracture",
        "second_fracture",
        "third_fracture",
        "fourth_fracture",
    ]:
        src = folder / f"{name}.txt"

        print(f"{src.name}:")
        # S is stored negative, so the net offset is M + S
        pts = read_points(src) - (M + S)
        print(f"  shifted points\n{pts}")
        tris = triangulate(pts)
        print(f"  {len(tris)} triangles\n{tris}")
        out = folder / f"{name}.stl"
        meshio.write(out, meshio.Mesh(pts, [("triangle", tris)]), binary=False)
        print(f"  written {out}")
