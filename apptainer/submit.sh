#!/usr/bin/env bash
# Submit a PBS job that runs one command inside the dtm_voxelization
# image through apptainer/run.sh, on one node with ncpus free cores.
#
#     apptainer/submit.sh [run.sh options] [--] command ...
#     apptainer/submit.sh -o /scratch/$USER/rialba -- bash pipeline.sh
#     apptainer/submit.sh dtm-voxelization cases/rialba.toml speed
#
# The arguments go to apptainer/run.sh unchanged, so relative paths in the
# command are relative to the repository (/workspace in the container), and
# relative paths in the options are relative to the folder you submit from.
# The environment at submission (APPTAINER_BIN, DTM_IMAGE, DTM_OUTPUT, ...) is
# exported to the job (qsub -V). The job log, stdout and stderr together, is
# written to logs/dtm_voxelization.<jobid>.log in the repository.
# Resources: DTM_NCPUS (default 32) and DTM_MEM_GB, memory in gigabytes
# (default 128, which is twice what a 2.5 M cell Rialba measured and leaves
# room for a bigger target_cells), e.g. DTM_MEM_GB=256 apptainer/submit.sh ...
# PBS does not confine the job to its cores, so the threads of the libraries
# (DTM_THREADS, see apptainer/shell.sh) are set to DTM_NCPUS unless given.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "${here}/.." && pwd)

case "${1:-}" in
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "")        echo "no command given" >&2; exit 1 ;;
esac

logs="${repo}/logs"
mkdir -p "${logs}"

# the job writes its own log (stdout and stderr) straight into logs/, so PBS
# does not have to copy a spool file back at the end; PBS_JOBID is expanded
# when the job runs. The job also starts in the home directory: go back to the
# submission folder
job="exec > $(printf '%q' "${logs}")/dtm_voxelization.\${PBS_JOBID}.log 2>&1; cd $(printf '%q' "${PWD}") && $(printf '%q ' "${here}/run.sh" "$@")"

ncpus="${DTM_NCPUS:-32}"
export DTM_THREADS="${DTM_THREADS:-${ncpus}}"
mem_gb="${DTM_MEM_GB:-128}"

echo "job:  ${job}"
echo "log:  ${logs}/dtm_voxelization.<jobid>.log"
echo "resources: ncpus=${ncpus}, mem=${mem_gb} GB, threads=${DTM_THREADS}"
echo "${job}" | qsub -V -N dtm_voxelization -l "select=1:ncpus=${ncpus}:mem=${mem_gb}gb" -j oe -o /dev/null
