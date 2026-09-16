"""Run a case: `python -m dtm_voxelization CASE.toml [STEP ...]`.

Steps always run in this order; with no STEP given, all of them:

  grid      refined, terrain-carved hex grid of the DTM -> grid.vtu
  surfaces  the fracture surfaces, aligned            -> surfaces/<name>.stl
  cut       voxelized image of each surface           -> cut_faces/<name>.npz
  detach    grid split along all cut faces at once    -> detached.vtu
  speed     SPEED mesh with labelled boundary quads   -> speed/<case>.mesh
"""

import sys
import time

from . import cut, detach, grid, speed, surfaces
from .case import load_case
from .mesh import peak_memory_gb

STEPS = {
    "grid": grid.run,
    "surfaces": surfaces.run,
    "cut": cut.run,
    "detach": detach.run,
    "speed": speed.run,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        raise SystemExit(__doc__)

    unknown = [s for s in argv[1:] if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}, choose from {list(STEPS)}")
    selected = [s for s in STEPS if s in argv[1:]] or list(STEPS)

    case = load_case(argv[0])
    print(f"case {case.name} from {argv[0]}")
    print(f"  dtm     {case.points}: {case.surface}, outward {case.outward}, "
          f"trim_to_footprint={case.trim_to_footprint}"
          + (f", voxel {case.voxel_size}, depth {case.poisson_depth}" if case.surface == "poisson" else ""))
    print(f"  output  {case.output}")
    print(
        f"  grid    target {case.target_cells:,} cells, inner regions {case.inner_regions}, "
        f"outer scale {case.outer_scale}, z padding {case.z_padding} / "
        f"{case.region_z_padding}, validate_mesh={case.validate_mesh}"
    )
    for s in case.surfaces:
        source = s.stl if s.stl is not None else f"{len(s.corners)} corners"
        print(f"  surface {s.name}: {source}")
    print(f"  steps   {selected}", flush=True)

    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        STEPS[name](case)
        print(
            f"=== {name} done in {time.perf_counter() - t0:.1f}s, "
            f"peak memory {peak_memory_gb():.2f} GB ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
