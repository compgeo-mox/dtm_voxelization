"""The railway through the case, as a path a train can run along:

    python -m dtm_voxelization.railway [CASE.toml]

The track is OpenStreetMap's (ODbL, (c) OpenStreetMap contributors): every
`railway=rail` way over the DTM's rectangle, fetched once from the Overpass API
and kept in data/xyz/railway_osm.json, so a rerun works offline and on the
cluster. At Rialba it is the Lecco-Tirano line, single track, which runs in
the open along the lake, under avalanche galleries in places, and in tunnels
at both ends of the domain.

The ways are chained end to end into one line, which has to come out as a
single piece -- a branch or a gap stops the run -- and clipped to the DTM's
rectangle. Its height is the terrain's where the track is in the open or
under a gallery and, in a tunnel, a straight grade between the terrain at the
two portals, PORTAL m out of them (flat at the one portal a tunnel reaching the edge of the domain
has). The DTM is ground with walls and cuttings in it, so the result is
smoothed over SMOOTH metres: a track does not climb a retaining wall.

Coordinates are the DTM's frame, UTM 32N minus the cloud's mean, the frame
of the SPEED meshes. Writes, in the case's output folder,

  railway.csv   s x y z kind, every STEP metres from the south-east end;
                kind 0 in the open, 1 under a gallery, 2 in a tunnel
  railway.vtu   the same as a line, with s and kind on it, for ParaView
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import meshio
import numpy as np
from shapely.geometry import LineString, box

if __package__:
    from .case import load_case
    from .dtm_io import _load_xyz, load_dtm_interpolator
    from .shift_fracture import M
else:  # run as a plain file, from an editor's Run button: no package around it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dtm_voxelization.case import load_case
    from dtm_voxelization.dtm_io import _load_xyz, load_dtm_interpolator
    from dtm_voxelization.shift_fracture import M

CASE = "cases/rialba.toml"  # when none is given, as from an editor's Run button
CACHE = "data/xyz/railway_osm.json"
OVERPASS = "https://overpass-api.de/api/interpreter"
STEP = 2.0  # m between the points of the path
SMOOTH = 60.0  # m, the window the height is averaged over
PORTAL = 20.0  # m outside a tunnel's portal where its grade is anchored
OPEN, GALLERY, TUNNEL = 0, 1, 2


def fetch(cache, lo, hi):
    """Overpass's railway ways over the rectangle lo..hi (DTM frame), cached."""
    if cache.exists():
        print(f"  {cache}: cached, not fetched again", flush=True)
        return json.loads(cache.read_text())
    from rasterio.warp import transform

    lon, lat = transform("EPSG:32632", "EPSG:4326", [lo[0] + M[0], hi[0] + M[0]],
                         [lo[1] + M[1], hi[1] + M[1]])
    query = (f'[out:json];way["railway"="rail"]'
             f"({lat[0]:.6f},{lon[0]:.6f},{lat[1]:.6f},{lon[1]:.6f});out geom tags;")
    print(f"  fetching {OVERPASS}: {query}", flush=True)
    request = urllib.request.Request(
        OVERPASS, data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "dtm_voxelization railway"})
    for attempt in range(1, 6):  # the public Overpass answers 502 when busy
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            print(f"  attempt {attempt}: {error}", flush=True)
            if attempt == 5 or error.code not in (429, 502, 503, 504):
                raise
            time.sleep(15 * attempt)
    cache.write_text(json.dumps(data, indent=1))
    print(f"  wrote {cache}: {len(data['elements'])} ways", flush=True)
    return data


def kind_of(tags):
    tunnel = tags.get("tunnel")
    if tunnel == "avalanche_protector" or tags.get("covered") == "yes":
        return GALLERY
    return TUNNEL if tunnel in ("yes", "building_passage") else OPEN


def chain(ways):
    """The ways joined end to end: (points (n, 2), kind of each segment (n-1,))."""
    from rasterio.warp import transform

    pieces = []
    for way in ways:
        east, north = transform("EPSG:4326", "EPSG:32632",
                                [p["lon"] for p in way["geometry"]],
                                [p["lat"] for p in way["geometry"]])
        xy = np.column_stack([east, north]) - M[:2]
        kind = kind_of(way.get("tags", {}))
        pieces.append((xy, kind))
        print(f"    way {way['id']}: {len(xy)} nodes, kind {kind} "
              f"({way.get('tags', {}).get('tunnel', 'open')}), "
              f"({xy[0, 0]:.0f},{xy[0, 1]:.0f}) to ({xy[-1, 0]:.0f},{xy[-1, 1]:.0f})", flush=True)

    line, kinds = pieces[0][0], [pieces[0][1]] * (len(pieces[0][0]) - 1)
    left = pieces[1:]
    while left:
        for i, (xy, kind) in enumerate(left):
            if np.allclose(xy[0], line[-1], atol=0.01):
                line, kinds = np.vstack([line, xy[1:]]), kinds + [kind] * (len(xy) - 1)
            elif np.allclose(xy[-1], line[-1], atol=0.01):
                line, kinds = np.vstack([line, xy[::-1][1:]]), kinds + [kind] * (len(xy) - 1)
            elif np.allclose(xy[-1], line[0], atol=0.01):
                line, kinds = np.vstack([xy[:-1], line]), [kind] * (len(xy) - 1) + kinds
            elif np.allclose(xy[0], line[0], atol=0.01):
                line, kinds = np.vstack([xy[::-1][:-1], line]), [kind] * (len(xy) - 1) + kinds
            else:
                continue
            left.pop(i)
            break
        else:
            raise SystemExit(f"{len(left)} railway ways do not join the line: a branch or a gap")
    return line, np.array(kinds)


def resample(line, kinds, lo, hi):
    """Points every STEP m along the line clipped to lo..hi, with their kind,
    from its south-east end."""
    if line[0, 1] > line[-1, 1]:
        line, kinds = line[::-1], kinds[::-1]
    edges = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    clipped = LineString(line).intersection(box(lo[0], lo[1], hi[0], hi[1]))
    if clipped.geom_type != "LineString":
        raise SystemExit(f"the railway crosses the domain {clipped.geom_type}, not in one piece")
    full = LineString(line)
    start, end = sorted(full.project(clipped.interpolate(d)) for d in (0, clipped.length))
    s = np.arange(start, end, STEP)
    xy = np.array([full.interpolate(d).coords[0] for d in s])
    kind = kinds[np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(kinds) - 1)]
    print(f"  railway: {full.length:.0f} m in all, {end - start:.0f} m in the domain, "
          f"{len(s):,} points every {STEP:g} m", flush=True)
    return s - start, xy, kind


def heights(s, xy, kind, terrain):
    """Terrain in the open and under galleries, a straight grade in tunnels,
    then averaged over SMOOTH m."""
    z = terrain(xy)
    inside = kind == TUNNEL
    runs = np.flatnonzero(np.diff(np.concatenate([[0], inside.astype(int), [0]])))
    # OSM puts a portal on the face of the slope, where the DTM is still the
    # hillside over the track: the grade is taken from the ground PORTAL m out
    out = int(round(PORTAL / STEP))
    for a, b in zip(runs[::2], runs[1::2]):
        before = max(a - 1 - out, 0) if a > 0 else None
        after = min(b + out, len(s) - 1) if b < len(s) else None
        ends = [i for i in (before, after) if i is not None]
        if not ends:
            raise SystemExit("the railway is in a tunnel over the whole domain")
        za, zb = z[ends[0]], z[ends[-1]]
        sa, sb = s[ends[0]], s[ends[-1]]
        z[a:b] = za if sa == sb else za + (s[a:b] - sa) * (zb - za) / (sb - sa)
        print(f"    tunnel s {s[a]:.0f}..{s[b - 1]:.0f} m: portal(s) at s "
              f"{', '.join(f'{s[i]:.0f}' for i in ends)}, z {za:.1f} -> {zb:.1f}", flush=True)
    half = int(round(SMOOTH / STEP / 2))
    padded = np.concatenate([np.full(half, z[0]), z, np.full(half, z[-1])])
    smooth = np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")
    open_ = ~inside
    print(f"  heights: smoothing over {SMOOTH:g} m moves the open track by "
          f"{np.abs(smooth - z)[open_].max():.1f} m at most, "
          f"{np.median(np.abs(smooth - z)[open_]):.2f} at the median; "
          f"z {smooth.min():.1f} .. {smooth.max():.1f}", flush=True)
    return smooth


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    case = load_case(argv[0] if argv else root / CASE)

    x, y, _ = _load_xyz(case.points)
    lo, hi = np.array([x.min(), y.min()]), np.array([x.max(), y.max()])
    print(f"  DTM rectangle {lo.round(1)} .. {hi.round(1)}", flush=True)
    data = fetch(root / CACHE, lo, hi)
    line, kinds = chain(data["elements"])
    s, xy, kind = resample(line, kinds, lo, hi)
    for k, name in [(OPEN, "open"), (GALLERY, "gallery"), (TUNNEL, "tunnel")]:
        print(f"    {name}: {STEP * (kind == k).sum():.0f} m", flush=True)

    z = heights(s, xy, kind, load_dtm_interpolator(case.points))
    case.output.mkdir(parents=True, exist_ok=True)
    table = case.output / "railway.csv"
    np.savetxt(table, np.column_stack([s, xy, z, kind]), fmt="%.3f %.3f %.3f %.3f %d",
               header="s x y z kind  (DTM frame; kind 0 open, 1 gallery, 2 tunnel; "
                      "track (c) OpenStreetMap contributors, ODbL)")
    ends = np.arange(len(s))
    vtu = case.output / "railway.vtu"
    meshio.write_points_cells(vtu, np.column_stack([xy, z]),
                              [("line", np.column_stack([ends[:-1], ends[1:]]))],
                              point_data={"s": s, "kind": kind.astype(float)})
    print(f"wrote {table} and {vtu}: {len(s):,} points", flush=True)


if __name__ == "__main__":
    main()
