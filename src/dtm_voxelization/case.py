"""Case description: everything that changes from one DTM to the next.

A case is a TOML file (see `cases/rialba.toml`); relative paths in it are
resolved against the file's own folder. Unknown or missing keys are errors.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Surface:
    """A fracture surface: an STL plus the translation into the DTM's frame,
    or the corners of a polygon given directly in that frame."""

    name: str
    stl: Path | None
    translation: tuple
    corners: np.ndarray | None


@dataclass(frozen=True)
class Case:
    name: str
    output: Path
    points: Path  # the DTM's point cloud
    surface: str  # "height_field" or "poisson"
    outward: tuple | None  # rotate the points' least-squares plane horizontal, this side up
    trim_to_footprint: bool  # domain = largest rectangle inside the points' footprint
    voxel_size: float | None  # poisson only
    poisson_depth: int | None  # poisson only
    target_cells: int
    z_padding: float
    inner_regions: tuple  # one or more (xmin, xmax, ymin, ymax) rectangles
    region_z_padding: float
    outer_scale: float
    validate_mesh: bool
    surfaces: tuple

    @property
    def frame_path(self):
        return self.output / "frame.npz"

    @property
    def grid_path(self):
        return self.output / "grid.vtu"

    @property
    def dtm_surface_path(self):
        return self.output / "dtm_surface.stl"

    @property
    def surface_dir(self):
        return self.output / "surfaces"

    @property
    def cut_dir(self):
        return self.output / "cut_faces"

    @property
    def detached_path(self):
        return self.output / "detached.vtu"

    @property
    def speed_dir(self):
        return self.output / "speed"


def _check_keys(table, required, optional, where):
    unknown = set(table) - set(required) - set(optional)
    missing = set(required) - set(table)
    if unknown or missing:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        )


def load_case(path):
    path = Path(path).resolve()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    root = path.parent

    _check_keys(data, {"name", "output", "dtm", "grid"}, {"surfaces"}, path.name)

    dtm = data["dtm"]
    where = f"{path.name} [dtm]"
    if dtm.get("surface") == "poisson":
        _check_keys(
            dtm,
            {"points", "surface", "trim_to_footprint", "voxel_size", "poisson_depth"},
            {"outward"},
            where,
        )
    elif dtm.get("surface") == "height_field":
        _check_keys(dtm, {"points", "surface", "trim_to_footprint"}, set(), where)
    else:
        raise ValueError(f"{where}: surface must be 'height_field' or 'poisson'")
    if "outward" in dtm and len(dtm["outward"]) != 3:
        raise ValueError(f"{where}: outward must be [x, y, z]")

    grid = data["grid"]
    grid_keys = {
        "target_cells",
        "z_padding",
        "inner_regions",
        "region_z_padding",
        "outer_scale",
        "validate_mesh",
    }
    _check_keys(grid, grid_keys, set(), f"{path.name} [grid]")
    regions = grid["inner_regions"]
    if not regions or any(len(r) != 4 for r in regions):
        raise ValueError(
            f"{path.name} [grid]: inner_regions is a list of [xmin, xmax, ymin, ymax]"
        )

    surfaces = []
    for i, entry in enumerate(data.get("surfaces", [])):
        where = f"{path.name} [[surfaces]] #{i + 1}"
        _check_keys(entry, {"name"}, {"stl", "translation", "corners"}, where)
        if ("stl" in entry) == ("corners" in entry):
            raise ValueError(f"{where}: give exactly one of 'stl' or 'corners'")
        if "corners" in entry:
            if "translation" in entry:
                raise ValueError(f"{where}: 'translation' applies to 'stl' only")
            corners = np.asarray(entry["corners"], dtype=float)
            if corners.ndim != 2 or corners.shape[1] != 3 or len(corners) < 3:
                raise ValueError(f"{where}: corners must be at least 3 [x, y, z] points")
            surfaces.append(Surface(entry["name"], None, (0.0, 0.0, 0.0), corners))
        else:
            translation = tuple(float(t) for t in entry.get("translation", (0, 0, 0)))
            if len(translation) != 3:
                raise ValueError(f"{where}: translation must be [dx, dy, dz]")
            surfaces.append(Surface(entry["name"], root / entry["stl"], translation, None))

    names = [s.name for s in surfaces]
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: surface names must be unique, got {names}")

    return Case(
        name=data["name"],
        output=root / data["output"],
        points=root / dtm["points"],
        surface=dtm["surface"],
        outward=tuple(float(v) for v in dtm["outward"]) if "outward" in dtm else None,
        trim_to_footprint=bool(dtm["trim_to_footprint"]),
        voxel_size=float(dtm["voxel_size"]) if "voxel_size" in dtm else None,
        poisson_depth=int(dtm["poisson_depth"]) if "poisson_depth" in dtm else None,
        target_cells=int(grid["target_cells"]),
        z_padding=float(grid["z_padding"]),
        inner_regions=tuple(tuple(float(v) for v in r) for r in regions),
        region_z_padding=float(grid["region_z_padding"]),
        outer_scale=float(grid["outer_scale"]),
        validate_mesh=bool(grid["validate_mesh"]),
        surfaces=tuple(surfaces),
    )
