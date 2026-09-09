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

## Data

`data/xyz/merged.xyz` is the raw DTM point cloud (`X Y Z` columns,
2-line header).

## Issues

Create an [issue](https://github.com/compgeo-mox/dtm_voxelization/issues).

## License

See [license](./LICENSE).