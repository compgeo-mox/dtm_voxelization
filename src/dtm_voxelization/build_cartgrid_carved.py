"""Pure Cartesian grid (PorePy's CartGrid, no local refinement at all --
just an anisotropic per-axis resolution chosen so the domain's real
extents come out close to isotropic cells) + carve away every cell
entirely above the real DTM surface.

Cell counts (NX, NY, NZ) are chosen so the CARVED result lands close to
TARGET_TOTAL_CELLS, based on the real terrain's own below-surface
fraction of the padded domain box (estimated once, cheaply, from a
coarse sample of the DTM interpolator).
"""

from pathlib import Path

import numpy as np
import porepy as pp

from .dtm_io import load_dtm_analysis_grid, load_dtm_interpolator

REPO_ROOT = Path(__file__).resolve().parents[2]
XYZ_PATH = REPO_ROOT / "data" / "xyz" / "merged.xyz"
OUTPUT_DIR = REPO_ROOT / "output"

TARGET_TOTAL_CELLS = 100_000
FRACTION_SAMPLE_N = (
    400  # resolution of the coarse pre-sample used only to size NX/NY/NZ
)


def estimate_kept_fraction(
    dtm_interp, xmin, xmax, ymin, ymax, zmin, lz, n=FRACTION_SAMPLE_N
):
    xs = xmin + (np.arange(n) + 0.5) * (xmax - xmin) / n
    ys = ymin + (np.arange(n) + 0.5) * (ymax - ymin) / n
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    terrain = dtm_interp(np.column_stack([xx.ravel(), yy.ravel()])).reshape(n, n)
    frac = np.clip((terrain - zmin) / lz, 0, 1)
    return float(np.mean(frac))


def main():
    dtm = load_dtm_analysis_grid(XYZ_PATH)
    XMIN, XMAX = dtm["xmin"], dtm["xmax"]
    YMIN, YMAX = dtm["ymin"], dtm["ymax"]
    ZMIN = float(np.nanmin(dtm["Z"])) - 50.0
    ZMAX = float(np.nanmax(dtm["Z"])) + 50.0
    Lx, Ly, Lz = XMAX - XMIN, YMAX - YMIN, ZMAX - ZMIN

    dtm_interp = load_dtm_interpolator(XYZ_PATH)
    kept_fraction = estimate_kept_fraction(dtm_interp, XMIN, XMAX, YMIN, YMAX, ZMIN, Lz)
    print(f"estimated below-terrain (kept) fraction of the box: {kept_fraction:.4f}")

    cell_edge = (Lx * Ly * Lz * kept_fraction / TARGET_TOTAL_CELLS) ** (1.0 / 3.0)
    NX = max(1, round(Lx / cell_edge))
    NY = max(1, round(Ly / cell_edge))
    NZ = max(1, round(Lz / cell_edge))
    print(
        f"cell edge target: {cell_edge:.3f}  ->  NX,NY,NZ = {NX},{NY},{NZ}  "
        f"(full grid {NX * NY * NZ:,} cells, estimated kept {NX * NY * NZ * kept_fraction:,.0f})"
    )

    g = pp.CartGrid(
        [NX, NY, NZ],
        physdims={
            "xmin": XMIN,
            "xmax": XMAX,
            "ymin": YMIN,
            "ymax": YMAX,
            "zmin": ZMIN,
            "zmax": ZMAX,
        },
    )

    print(f"built CartGrid: {g.num_cells:,} cells, {g.num_nodes:,} nodes")

    # Cell centers computed explicitly from the known uniform spacing, rather
    # than via g.compute_geometry() -- much cheaper for a large Cartesian grid.
    # Cell numbering is x-fastest, then y, then z (matches CartGrid's own).
    hx, hy, hz = Lx / NX, Ly / NY, Lz / NZ
    xs = XMIN + (np.arange(NX) + 0.5) * hx
    ys = YMIN + (np.arange(NY) + 0.5) * hy
    zs = ZMIN + (np.arange(NZ) + 0.5) * hz
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.vstack(
        [xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")]
    )  # shape (3, num_cells)

    terrain_at_center = dtm_interp(np.column_stack([centers[0], centers[1]]))
    cell_top = centers[2] + 0.5 * hz
    keep = cell_top <= terrain_at_center  # entirely-above-terrain cells are carved away
    print(
        f"kept cells: {int(keep.sum()):,} / {g.num_cells:,} "
        f"({100.0 * keep.sum() / g.num_cells:.1f}%)"
    )

    carved_g, _, _ = pp.partition.extract_subgrid(g, keep)
    print(f"extracted carved grid: {carved_g.num_cells:,} cells")

    OUTPUT_DIR.mkdir(exist_ok=True)
    exporter = pp.Exporter(carved_g, "cartgrid_carved", folder_name=str(OUTPUT_DIR))
    exporter.write_vtu()
    print(f"wrote {OUTPUT_DIR / 'cartgrid_carved_2.vtu'} (carved grid only)")


if __name__ == "__main__":
    main()
