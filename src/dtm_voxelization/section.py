"""The mesh's profile on the plane that stands normal to the case's fracture
surfaces: `python -m dtm_voxelization.section [CASE.toml]`, or the file itself
from an editor, which takes CASE below.

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

Cutting the mesh with it gives the profile; the surfaces are cut with the same
plane, so their traces lie in it too, and the plane itself is written as a
rectangle over the mesh, which is what makes the three readable together in
ParaView.

Written to the case's output as section.vtp, section_surfaces.vtp and
section_plane.vtp.
"""

import sys
from pathlib import Path

import meshio
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter, vtkXMLUnstructuredGridReader

if __package__:
    from .case import load_case
    from .clean_stl import to_polydata
    from .mesh import require
    from .surfaces import in_grid_frame
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.case import load_case
    from dtm_voxelization.clean_stl import to_polydata
    from dtm_voxelization.mesh import require
    from dtm_voxelization.surfaces import in_grid_frame

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button


def through_the_fractures(case):
    """(normal, origin) of the vertical plane threading the fractures' middles."""
    centres, normals = [], []
    for surface in case.surfaces:
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
    for surface, centre, mean in zip(case.surfaces, centres, normals):
        print(
            f"    {surface.name}: centre {abs((centre - origin) @ normal):.1f} m off the plane, "
            f"met at {90 - np.degrees(np.arcsin(abs(mean @ normal))):.0f} degrees"
        )
    return normal, origin


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


def cut(data, plane, path):
    """Write the section of one dataset by the plane."""
    cutter = vtkCutter()
    cutter.SetInputData(data)
    cutter.SetCutFunction(plane)
    cutter.Update()
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(cutter.GetOutput())
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToLZ4()
    if not writer.Write():
        raise RuntimeError(f"vtkXMLPolyDataWriter failed on {path}")
    out = cutter.GetOutput()
    print(f"wrote {path}: {out.GetNumberOfCells():,} cells, {out.GetNumberOfPoints():,} points")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)
    if not case.surfaces:
        raise SystemExit(f"{case.name}: no surfaces, so no plane to stand normal to")

    normal, origin = through_the_fractures(case)
    plane = vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    mesh = case.detached_path if case.detached_path.exists() else case.grid_path
    require(mesh, "grid")
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(mesh))
    reader.Update()
    print(f"  cutting {mesh}: {reader.GetOutput().GetNumberOfCells():,} cells")
    cut(reader.GetOutput(), plane, case.output / "section.vtp")

    corners = np.vstack([in_grid_frame(case, s).reshape(-1, 3) for s in case.surfaces])
    faces = np.arange(len(corners)).reshape(-1, 3)
    cut(to_polydata(corners, faces), plane, case.output / "section_surfaces.vtp")

    write_plane(case.output / "section_plane.vtp", normal, origin, reader.GetOutput().GetBounds())


if __name__ == "__main__":
    main()
