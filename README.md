[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# dtm_voxelization

Refined, terrain-carved hex grids of a real DTM (digital terrain model), split
along fracture surfaces and exported to [SPEED](https://bitbucket.org/ilmaz/speed).

A **case** -- one DTM, its refinement region and its fracture surfaces -- is a
TOML file. Running it goes through five steps, each reading the previous one's
output from disk:

| step | does | writes (in the case's `output`) |
|---|---|---|
| `grid` | Cartesian hex grid over the DTM with two nested refined regions, carved: hexes not in the rock are dropped | `grid.vtu`, `frame.npz`, `dtm_surface.stl` |
| `surfaces` | fracture surfaces as triangles in the DTM's frame, for inspection | `surfaces/<name>.stl` |
| `cut` | the voxelized image of each surface on the grid | `cut_faces/<name>.npz`, `.vtu` |
| `detach` | the grid split along the cut faces of all surfaces at once | `detached.vtu` |
| `speed` | SPEED mesh with labelled boundary quads | `speed/<case>.mesh`, `speed/<case>_boundary.vtu` |

## Installation

Python >= 3.11, with numpy, scipy and meshio:

```bash
pip install -e .
```

Cases with `surface = "poisson"` also need Open3D, and turning a LAS point
cloud into the inputs below needs laspy and VTK:

```bash
pip install -e ".[poisson]"
pip install -e ".[las]"
```

## Usage

```bash
python -m dtm_voxelization cases/rialba.toml                   # all steps
python -m dtm_voxelization cases/rialba.toml cut detach speed  # only these
```

(`dtm-voxelization` once installed.) Steps always run in pipeline order,
whatever order they are given in. Rerunning only the later ones is the usual
way to change the surfaces without rebuilding a large grid.

A LAS point cloud becomes a case's `points` file, plus two files for
ParaView, all written next to it:

```bash
python -m dtm_voxelization.las_export path/to/cloud.las cases/san_martino.toml   # cloud.xyz, .vtp, .stl
```

All three are in the cases' frame: x/y centred on the cloud's mean (rounded
to 1 mm, logged and written in the `.xyz` comment line), z elevation. The
`.vtp` holds every point in binary -- open it rather than the `.xyz`, whose
size ParaView's CSV reader does not survive. The `.stl` is the surface the
given `poisson` case would reconstruct from the cloud, with its `outward`,
`voxel_size` and `poisson_depth`, cut down to the triangles within two octree
cells of a point.

## A case

A new DTM is a new copy of [`cases/rialba.toml`](cases/rialba.toml) (a
terrain) or [`cases/san_martino.toml`](cases/san_martino.toml) (a wall with
overhangs). Relative paths are resolved against the TOML file's folder;
unknown or missing keys are errors.

```toml
name = "rialba"                  # also names the SPEED mesh: speed/rialba.mesh
output = "../output/rialba"

[dtm]
points = "../data/xyz/merged.xyz"   # point cloud, X Y Z columns, 2-line header
surface = "height_field"            # the terrain is z = f(x, y)
trim_to_footprint = false           # the points cover the whole rectangle

[grid]
target_cells = 100_000           # carved cells to aim for (accepted within +-30%)
z_padding = 50.0                 # domain below the lowest and above the highest DTM point
inner_regions = [                # level-2 rectangles: xmin, xmax, ymin, ymax
    [-260.0, -110.0, 175.0, 270.0],
    [-200.0,  -60.0, 240.0, 360.0],
]
region_z_padding = 20.0          # clearance above and below the terrain inside each
outer_scale = 1.5                # level-1 region: the inner one scaled in x, y and z,
                                 # never less than 1.5 coarse cells wider in x and y
validate_mesh = false            # exhaustive conformity checks, slow on large grids

[[surfaces]]                     # an STL, moved into the DTM's frame
name = "frattura_verticale"
stl = "../data/surfaces/frattura_verticale_shifted.stl"
translation = [0.0, 0.0, -474.976610]

[[surfaces]]                     # or a polygon given directly in that frame
name = "piano_1"
corners = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]]
```

`surfaces` may be left out entirely: the grid is then exported to SPEED
without cracks.

A DTM that is not a function of (x, y) -- a wall, overhangs, re-entrances --
takes a reconstructed surface instead:

```toml
[dtm]
points = "../data/xyz/san_martino.xyz"
surface = "poisson"              # a 3D surface reconstructed from the points
outward = [0.0, 1.0, 0.0]        # rotate the least-squares plane horizontal, out of the rock up
trim_to_footprint = true         # domain: largest rectangle inside the points' footprint
voxel_size = 0.25                # down-sampling before the reconstruction
poisson_depth = 10               # octree depth of the reconstruction
```

With `outward` the grid is built in the rotated frame, so `inner_regions` are
given in that frame too: the `grid` step logs the domain rectangle it builds.
Fracture surfaces stay in the DTM's own frame and are rotated for you.

The terrain surface is written to `dtm_surface.stl` (binary) in the grid's
frame, so it overlays `grid.vtu`, `detached.vtu` and `cut_faces/` in
ParaView; for a `poisson` surface only the part over the domain rectangle is
kept. The SPEED mesh and its `_boundary.vtu` are in the DTM's frame instead.

## Method

**Terrain.** A `height_field` is the linear interpolant of the points over
their Delaunay triangulation in (x, y), and a hex is kept when its top lies
below it at the hex's (x, y) centroid. A `poisson` surface can fold over
itself: the points are down-sampled, their normals estimated and oriented
consistently along the surface (so the underside of an overhang points
down), and a screened Poisson reconstruction (Open3D) gives a triangle mesh.
A hex is then kept when the vertical ray up from the top of its centroid's
segment crosses the mesh an odd number of times -- the top is in the rock --
and the segment down to the hex bottom crosses nothing. On a height field
this is the same rule. With `outward`, all of it happens in the frame where
the points' total least-squares plane is horizontal; the rotation is saved
in `frame.npz`.

**Grid.** The inner regions are split twice (3x3x3, then 3x3x3 again), the
outer region -- their bounding box scaled by `outer_scale`, and in x/y at least 1.5
coarse cells wider on every side, which the second split's buffer needs -- once, with the
conforming wall/corner/concave transition templates of
[`templates.py`](src/dtm_voxelization/templates.py) around each. The coarse
cell size is first chosen without building anything, from the predicted
carved count -- coarse cells plus the hexes each refined level adds, weighted
by the rock fraction of its region -- so that even the first build is near
`target_cells`; builds then rescale it by the observed carved count until it
lands within 30%. The log reports the peak memory after each stage.

Each inner region takes its own z range from the terrain over its own
rectangle, so a slope needs no single box tall enough for all of it.
Regions at different depths are refined by separate calls, which must stay 3
cells of the first split apart -- one coarse cell -- since a column can take
only one transition role over its whole height. Closer ones, intersecting
ones included, are merged into one call over the union of their footprints,
as deep as the deepest; the log says when. Where that union's outline steps,
it must step by at least 2 cells: a reentrant corner with a single cell of
run past it has no transition template, and the build refuses it.

**Cut.** A grid face belongs to a surface's voxelized image when the segment
joining the two cells that share it crosses the surface an odd number of
times. Parity is what makes the result a continuous staircase sheet rather
than a fuzzy band. Ties -- a surface lying exactly on cell centres, which is
the normal case for a flat sheet -- are broken by nudging the surface once
along a direction skew to the grid. Surfaces may be curved, need not be
aligned with the grid and may end inside it.

**Detach.** Each node on a crack gets one copy per group of incident cells
that can still reach each other without crossing a cut face. The two sides
of a crack stop sharing points, while a node at a crack tip -- where the
cells wrap around the end of the surface -- stays one point, so cracks close
at their tips. All surfaces are split together, so crossing fractures come
out right. The split fails loudly if cells across a cut face still share a
node away from a tip. In `detached.vtu`, *Warp By Vector* on `opening` opens
the cracks.

**SPEED.** The `GRIDFILE` layout: nodes, then boundary quads, then hexes,
1-based. Hexes are in VTK/Exodus order with positive Jacobians and quads are
wound outward; both are checked before writing. Quad tags, for the `.mate`
file: `2` lateral and bottom faces (bounding-box planes), `3` everything else
on the boundary -- the carved terrain and both sides of every crack. Hexes
carry tag `1`. Tags are decided in the grid's frame; the mesh is written back
in the DTM's frame.

## Running in a container (Apptainer)

The image carries the dependencies, Open3D included (Python 3.12), not the code: `shell.sh` bind-mounts this
repository at `/workspace`, so edits on the host take effect without a
rebuild.

```bash
source apptainer/build.sh                      # -> apptainer/dtm_voxelization.sif
source apptainer/shell.sh                      # a shell inside it, repo at /workspace
dtm-voxelization cases/rialba.toml             # inside
```

The output folder is separately bindable, for writing results to scratch:

```bash
source apptainer/shell.sh -o /scratch/$USER/run1                       # /workspace/output -> there
source apptainer/shell.sh -- dtm-voxelization cases/rialba.toml speed  # one command and exit
source apptainer/shell.sh -b /scratch:/scratch                         # extra bind, repeatable
```

`-d` does the same for `data/`, `-a` (or `$APPTAINER_BIN`) picks the apptainer
executable, falling back to `/opt/mox/apptainer/bin/apptainer`. The container
runs with `--containall --no-home --writable-tmpfs`: only the binds are
visible, and results must land in one of them to survive. See the header of
[`build.sh`](apptainer/build.sh) for the cluster knobs (`APPTAINER_TMPDIR`,
`APPTAINER_CACHEDIR`, `--fakeroot`).

## Data

- `data/xyz/merged.xyz` -- the Rialba DTM point cloud, UTM32N minus its mean
  (527837.392605, 5082191.470488, 474.976610), z included.
- `data/surfaces/frattura_verticale_shifted.stl` -- a vertical fracture, x/y
  already in that frame, z still absolute (hence the case's translation).
- `data/xyz/san_martino.xyz` -- the San Martino wall from photogrammetry,
  818,623 points: UTM32N minus (530019.39293486, 5079649.72368661) in x/y,
  z elevation.

## Issues

Create an [issue](https://github.com/compgeo-mox/dtm_voxelization/issues).

## License

See [license](./LICENSE).
