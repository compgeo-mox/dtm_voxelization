[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# dtm_voxelization: voxelizing real terrain into PorePy grids

dtm_voxelization builds a carved [PorePy](https://github.com/pmgbergen/porepy)
Cartesian grid out of a real DTM (digital terrain model): a plain
`CartGrid` with an anisotropic per-axis resolution chosen so the
domain's real extents come out close to isotropic cells, then every
cell entirely above the true terrain surface is carved away.

**Key Features:**
- Automatic per-axis (NX, NY, NZ) sizing so the *carved* result lands
  close to a target total cell count
- Sizing based on the real terrain's own below-surface fraction of the
  padded domain box, estimated cheaply from a coarse sample of the DTM
- Full-resolution scattered-point DTM interpolation (no intermediate
  regular-grid resampling) when carving
- Seamless integration with [PorePy](https://github.com/pmgbergen/porepy)

## Installation for Linux

dtm_voxelization requires Python >= 3.10.

Since dtm_voxelization depends on [PorePy](https://github.com/pmgbergen/porepy),
we assume that the latter is accessible in your PYTHONPATH (it is also
declared as a dependency below, so a plain install below will normally
pull it in on its own).
To install dtm_voxelization
```bash
pip install -e .
```
avoid the `-e` if you do not want the editable version.

## Usage

```bash
dtm-voxelization
```

(equivalently: `python -m dtm_voxelization.build_cartgrid_carved`)

Reads the point cloud at `data/xyz/merged.xyz`, builds the carved
Cartesian grid, and writes the result as VTU files under `output/`.

Two constants at the top of
[`build_cartgrid_carved.py`](src/dtm_voxelization/build_cartgrid_carved.py)
control the run:

- `TARGET_TOTAL_CELLS` -- desired number of cells in the carved grid.
- `FRACTION_SAMPLE_N` -- resolution of the coarse pre-sample used only
  to estimate the terrain's below-surface fraction (does not affect
  the final grid resolution directly).

## Exporting planes

```bash
dtm-export-planes
```

(equivalently: `python -m dtm_voxelization.export_planes_stl`)

Writes one STL per polygon listed in `PLANES` at the top of
[`export_planes_stl.py`](src/dtm_voxelization/export_planes_stl.py), under
`output/planes/`. A polygon is just its corner points, in order around the
boundary; adding a new plane means appending one more entry. Corners given
as `(x, y)` make a horizontal polygon at the elevation `z` (default 0);
corners given as `(x, y, z)` are used as-is. Nothing else is read -- the
DTM plays no part here.

## Importing an external surface

```bash
dtm-align-surface
```

`data/xyz/merged.xyz` is not in map coordinates: `to_xyz.py` built it from
the UTM32N DTM by subtracting the mean of all three coordinate columns, z
included. A surface built elsewhere therefore has to be translated before
it can meet the grid --
[`align_surface.py`](src/dtm_voxelization/align_surface.py) lists each
external STL with the translation that lands it in this frame and writes
the aligned copy into `output/planes/`. The untransformed surfaces are
tracked under `data/surfaces/`, so the pipeline runs on any checkout --
a cluster, say -- without the folder they were originally built in.

## Voxelizing a surface onto the grid

```bash
dtm-cut-surface
```

(equivalently: `python -m dtm_voxelization.cut_surface`)

Reads the carved grid from `output/cartgrid_carved_refined.vtu` and every
STL under `output/planes/`, and computes for each one the *voxelized* image
of the surface on the grid: the set of existing grid faces that the surface
cuts through, written to `output/cut_faces/` as a quad mesh (`.vtu`, for
ParaView) and as `face_nodes` / `cell_pairs` arrays (`.npz`, the input to
the cell-detachment step).

A face is in the cut set when the segment joining the two cells that share
it crosses the STL an odd number of times. Parity is what makes the result
a continuous staircase surface instead of a fuzzy band, and it is exactly
the set of faces where the two cells must stop sharing nodes once the sides
are detached. The surface may be curved and need not be aligned with the
grid; it may also terminate inside the grid, in which case the staircase
simply ends at the tip. See
[`cut_surface.py`](src/dtm_voxelization/cut_surface.py) for the tie-breaking
rules that keep the sheet connected when it lies exactly on the cell centres.

## Detaching the two sides

```bash
dtm-detach-cells
```

Splits the grid along a cut-face set from `output/cut_faces/`: the cells
stay exactly where they are, but the two sides stop sharing points, so
they become logically disconnected. Each crack node is duplicated once per
group of incident cells that can still reach each other without crossing
the crack -- which keeps the crack TIP welded, where the cells wrap around
the end of the surface, instead of tearing the mesh open to the boundary.

The result goes to `output/detached/<name>_detached.vtu` with three fields
for checking it:

- `side_color` -- the cells around the crack, coloured by what they can
  still reach through shared nodes (ignoring the tip rim). Two colours
  that never mix across the surface means the sides really are separated.
- `crack_side` -- the same cells labelled -1/+1 from the STL geometry, an
  independent cross-check of the colouring.
- `opening` -- a point displacement. *Warp By Vector* on it in ParaView
  opens the crack, which is only possible because its nodes are now
  distinct points.

See [`detach_cells.py`](src/dtm_voxelization/detach_cells.py) for the
checks `main` runs on every split.

## Running in a container (Apptainer)

Built for clusters where Apptainer is the only container runtime:

```bash
source apptainer/build.sh   # -> apptainer/dtm_voxelization.sif
source apptainer/shell.sh   # a shell inside it, repo at /workspace
```

Both scripts work either way, sourced or run (`bash apptainer/shell.sh`).

The image carries the dependencies (PoRePy, cloned from GitHub since it is
not on PyPI, plus numpy/scipy/meshio in a venv at `/opt/venv`); it does NOT
carry the code. `shell.sh` bind-mounts this repository at `/workspace`, so
edits on the host take effect with no rebuild, and puts the same console
scripts on PATH as a local install (`dtm-cut-surface` and friends, with
`python -m dtm_voxelization.<module>` always available too).

The output folder is separately bindable, for writing results to scratch
rather than into the repository:

```bash
source apptainer/shell.sh -o /scratch/$USER/run1   # /workspace/output -> there
source apptainer/shell.sh -- dtm-cut-surface       # run one command and exit
source apptainer/shell.sh -b /scratch:/scratch     # extra bind, repeatable

export DTM_OUTPUT=/scratch/$USER/run1              # or set it once, then
source apptainer/shell.sh
```

PoRePy's `@njit(cache=True)` cannot write its cache into a read-only image,
so the image sets `NUMBA_CACHE_DIR=/tmp/numba_cache`. That is a fresh tmpfs
each session, meaning numba recompiles those functions on the first import;
point it at a bind-mounted folder to keep the cache warm:

```bash
export APPTAINERENV_NUMBA_CACHE_DIR=/workspace/.numba_cache
source apptainer/shell.sh
```

`-d` does the same for the input `data/` folder, `-a` (or `$APPTAINER_BIN`)
picks a specific apptainer executable when it is not on PATH -- it falls back
to `/opt/mox/apptainer/bin/apptainer`. The container runs with
`--containall --no-home --writable-tmpfs`, so nothing of the host is visible
beyond the binds and writes inside the image land in a throwaway overlay:
results have to go to `/workspace/output` (or another bind) to survive. See
[`dtm_voxelization.def`](apptainer/dtm_voxelization.def) for the build, and
the header of [`build.sh`](apptainer/build.sh) for the cluster knobs
(`APPTAINER_TMPDIR`, `APPTAINER_CACHEDIR`, `--fakeroot`).

## Data

`data/xyz/merged.xyz` is the raw DTM point cloud (`X Y Z` columns,
2-line header).

## Issues

Create an [issue](https://github.com/compgeo-mox/dtm_voxelization/issues).

## License

See [license](./LICENSE).