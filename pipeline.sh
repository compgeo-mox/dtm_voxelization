#!/usr/bin/env bash
# Everything one case needs, in order, so that a cluster run is a single job:
#
#     bash pipeline.sh [CASE.toml]
#     apptainer/submit.sh -- bash pipeline.sh
#
# The default case is cases/rialba.toml, and paths are relative to the
# repository, which is where the container starts. The steps are
#
#   area CASE                      the wall's surface, cut to the window
#   deposit_regions CASE           the two deposit outlines, from what they are drawn on
#   dtm-voxelization CASE          grid, surfaces, cut, detach, speed
#   source_points CASE             the surveyed sources, dropped on the grid
#   source_mesh CASE               the box around them, cut out of the grid
#   section CASE                   the profile, the fractures and the planes
#
# in that order because each one reads what the ones before it wrote: nothing
# here rebuilds the grid, so a failure half way is resumed by running what is
# left by hand. Each step is timed and the whole thing stops at the first one
# that fails, with its exit status.
#
# The first runs only when the case draws its surface from an `stl` that is
# not there yet -- on a fresh clone of a wall case, where the reconstruction has
# never been made -- the second only when the case carries a back deposit,
# whose outlines it writes back into data/regions/ before anything reads them,
# and the last three only when the case has what they need:
# surveyed sources for two of them, fracture surfaces for the section. So a
# case like San Martino, a bare wall, goes through the same one command and
# comes out with the same archive.
#
# Everything written then lands in one archive, named after the job (PBS_JOBID)
# when there is one and after the hour otherwise, so a run is a single file to
# copy off the cluster.

set -euo pipefail

case_file="${1:-cases/rialba.toml}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${here}"

echo "pipeline: ${case_file}"
echo "repo:     ${here}"
echo "python:   $(command -v python)"
echo

step() {
    local name="$1"
    shift
    echo "=== ${name}  $(date '+%H:%M:%S')"
    local started
    started=$(date +%s)
    "$@"
    echo "=== ${name} done in $(( $(date +%s) - started ))s"
    echo
}

# what the case holds, and so which steps have anything to do: the output
# folder, whether there are source points, whether there are surfaces
read -r output sources surfaces wall from_file regions <<< "$(python -c "import sys
from dtm_voxelization.case import load_case
case = load_case(sys.argv[1])
print(case.output, case.source_points is not None, bool(case.surfaces),
      case.stl or '-', case.stl is not None, case.back_deposit is not None)" "${case_file}")"
echo "output:   ${output}"
echo "source points: ${sources}, surfaces: ${surfaces}, surface from a file: ${from_file}"
echo "deposit outlines to draw: ${regions}"
echo

if [ "${from_file}" = "True" ] && [ ! -f "${wall}" ]; then
    step "surface" python -m dtm_voxelization.area "${case_file}"
fi

if [ "${regions}" = "True" ]; then
    step "regions" python -m dtm_voxelization.deposit_regions "${case_file}"
else
    echo "=== regions: the case carries no back deposit, skipped"
    echo
fi

step "pipeline" dtm-voxelization "${case_file}"
if [ "${sources}" = "True" ]; then
    step "source points" python -m dtm_voxelization.source_points "${case_file}"
    step "source mesh" python -m dtm_voxelization.source_mesh "${case_file}"
else
    echo "=== source points and source mesh: the case has none, skipped"
    echo
fi
if [ "${surfaces}" = "True" ]; then
    step "section" python -m dtm_voxelization.section "${case_file}"
else
    echo "=== section: the case has no surfaces to stand normal to, skipped"
    echo
fi

archive="$(dirname "${output}")/results_${PBS_JOBID:-$(date '+%Y%m%d_%H%M')}.tar.gz"
step "archive" tar -czf "${archive}" \
    -C "$(dirname "${output}")" "$(basename "${output}")"
echo "archive: ${archive}  ($(du -h "${archive}" | cut -f1))"

echo "all done  $(date '+%H:%M:%S')"
