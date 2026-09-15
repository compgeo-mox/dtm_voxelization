"""The grid's frame: the DTM's points rotated so their least-squares plane is
horizontal.

A wall is not a terrain, but rotated onto its least-squares plane it becomes
one: the plane is fitted by total least squares (the normal is the covariance
eigenvector with the smallest eigenvalue -- z = ax + by + c would degenerate
for a sub-vertical face), its normal is turned towards the case's `outward`
direction (out of the rock), and the in-plane axes follow the other two
eigenvectors so that the domain box is tight. The rotation is proper (det +1):
a reflection would turn every hex inside out when mapped back.

Everything from the grid to the detached grid lives in this frame; only the
SPEED export maps back to the DTM's frame. Without `outward` the frame is the
identity and points pass through untouched.
"""

import numpy as np


def compute(points, outward):
    """(rotation (3, 3), center (3,)) with rows of `rotation` = grid axes."""
    if outward is None:
        return np.eye(3), np.zeros(3)
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - center).T))
    normal, first = vectors[:, 0], vectors[:, 2]
    if normal @ np.asarray(outward, dtype=float) < 0:
        normal = -normal
    if first[np.argmax(np.abs(first))] < 0:
        first = -first
    rotation = np.vstack([first, np.cross(normal, first), normal])
    return rotation, center


def is_identity(rotation, center):
    return np.array_equal(rotation, np.eye(3)) and not center.any()


def to_grid(points, rotation, center):
    if is_identity(rotation, center):
        return points
    return (points - center) @ rotation.T


def to_dtm(points, rotation, center):
    if is_identity(rotation, center):
        return points
    return points @ rotation + center


def save(path, rotation, center):
    np.savez(path, rotation=rotation, center=center)


def load(path):
    data = np.load(path)
    return data["rotation"], data["center"]
