"""Hex-mesh helpers shared by the pipeline steps."""

import resource
from pathlib import Path

import meshio
import numpy as np

# VTK hexahedron: nodes 0-3 bottom face, 4-7 top face, same rotational order;
# each face wound so that its normal points out of the cell
HEX_FACES = np.array(
    [
        [0, 3, 2, 1],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
)


def peak_memory_gb():
    """Peak resident memory of this process so far."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def require(path, step):
    """Fail loudly when an input of a step is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run the '{step}' step first")


def read_hex_mesh(path):
    """VTU -> (points (N, 3) float, hexes (M, 8) int)."""
    mesh = meshio.read(str(path))
    hexes = np.vstack([b.data for b in mesh.cells if b.type == "hexahedron"])
    return mesh.points.astype(float), hexes


def build_face_table(hexes):
    """All distinct faces of a hex mesh.

    Returns (face_nodes, owners): `face_nodes` (F, 4) keeps the winding of
    the face's first owner, `owners` (F, 2) holds the two cells sharing it
    (second entry -1 on a boundary face)."""
    faces = hexes[:, HEX_FACES].reshape(-1, 4)
    cells = np.repeat(np.arange(len(hexes)), len(HEX_FACES))

    keys = np.sort(faces, axis=1)
    _, first, inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()

    if counts.max() > 2:
        raise ValueError(
            f"non-manifold mesh: {int((counts > 2).sum())} faces shared by more "
            "than two cells"
        )

    owners = np.full((len(counts), 2), -1, dtype=np.int64)
    # ordering by (face, cell) puts a face's owners next to each other, the
    # lower cell id first
    order = np.lexsort((cells, inverse))
    face_sorted, cell_sorted = inverse[order], cells[order]
    slot = np.zeros(len(face_sorted), dtype=int)
    slot[1:] = face_sorted[1:] == face_sorted[:-1]
    owners[face_sorted, slot] = cell_sorted

    return faces[first], owners
