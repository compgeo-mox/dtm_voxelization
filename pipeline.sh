#!/usr/bin/env bash
# Everything one case needs, in order, so that a cluster run is a single job:
#
#     bash pipeline.sh [CASE.toml]
#     apptainer/submit.sh -- bash pipeline.sh
#
# The default case is cases/rialba.toml, and paths are relative to the
# repository, which is where the container starts. The steps are
#
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

step "pipeline" dtm-voxelization "${case_file}"
step "source points" python -m dtm_voxelization.source_points "${case_file}"
step "source mesh" python -m dtm_voxelization.source_mesh "${case_file}"
step "section" python -m dtm_voxelization.section "${case_file}"

# the case says where it writes; the survey's own folder takes the source
# points, so both go in
output=$(python -c "import sys
from dtm_voxelization.case import load_case
print(load_case(sys.argv[1]).output)" "${case_file}")
archive="$(dirname "${output}")/results_${PBS_JOBID:-$(date '+%Y%m%d_%H%M')}.tar.gz"
step "archive" tar -czf "${archive}" \
    -C "$(dirname "${output}")" "$(basename "${output}")" \
    -C "${here}" data/source_pts
echo "archive: ${archive}  ($(du -h "${archive}" | cut -f1))"

echo "all done  $(date '+%H:%M:%S')"
