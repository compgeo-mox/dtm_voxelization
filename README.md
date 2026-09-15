[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# dtm_voxelization

Refined, terrain-carved hex grids of a real DTM (digital terrain model), split
along fracture surfaces and exported to [SPEED](https://bitbucket.org/ilmaz/speed).

A **case** -- one DTM, its refinement region and its fracture surfaces -- is a
TOML file. Running it goes through five steps, each reading the previous one's
output from disk:

| step | does | writes (in the case's `output`) |
|---|---|---|
| `grid` | Cartesian hex grid over the DTM with two nested refined regions, carved: hexes entirely above the terrain are dropped | `grid.vtu`, `dtm_surface.stl` |
| `surfaces` | fracture surfaces as triangles in the DTM's frame, for inspection | `surfaces/<name>.stl` |
| `cut` | the voxelized image of each surface on the grid | `cut_faces/<name>.npz`, `.vtu` |
| `detach` | the grid split along the cut faces of all surfaces at once | `detached.vtu` |
| `speed` | SPEED mesh with labelled boundary quads | `speed/<case>.mesh`, `speed/<case>_boundary.vtu` |

## Installation

Python >= 3.11, with numpy, scipy and meshio:

```bash
pip install -e .
```

## Usage

```bash
python -m dtm_voxelization cases/rialba.toml                   # all steps
python -m dtm_voxelization cases/rialba.toml cut detach speed  # only these
```

(`dtm-voxelization` once installed.) Steps always run in pipeline order,
whatever order they are given in. Rerunning only the later ones is the usual
way to change the surfaces without rebuilding a large grid.

## A case

A new DTM is a new copy of [`cases/rialba.toml`](cases/rialba.toml). Relative
paths are resolved against the TOML file's folder; unknown or missing keys
are errors.

```toml
name = "rialba"                  # also names the SPEED mesh: speed/rialba.mesh
dtm = "../data/xyz/merged.xyz"   # point cloud, X Y Z columns, 2-line header
output = "../output/rialba"

[grid]
target_cells = 100_000           # carved cells to aim for (accepted within +-30%)
z_padding = 50.0                 # domain below the lowest and above the highest DTM point
inner_region = [-260.0, -10.0, 175.0, 425.0]   # level-2 rectangle: xmin, xmax, ymin, ymax
region_z_padding = 20.0          # clearance above and below the terrain inside it
outer_scale = 1.5                # level-1 region: the inner one scaled in x, y and z
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

## Method

**Grid.** The inner region is split twice (3x3x3, then 3x3x3 again), the
outer region -- the inner one scaled by `outer_scale` -- once, with the
conforming wall/corner/concave transition templates of
[`templates.py`](src/dtm_voxelization/templates.py) around each. The coarse
cell size starts from a plain-grid estimate and is rescaled by the observed
carved count until it lands within 30% of `target_cells`.

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
carry tag `1`.

## Running in a container (Apptainer)

The image carries the dependencies, not the code: `shell.sh` bind-mounts this
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

## Issues

Create an [issue](https://github.com/compgeo-mox/dtm_voxelization/issues).

## License

See [license](./LICENSE).
