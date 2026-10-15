#!/usr/bin/env bash
# Run one command inside the dtm_voxelization image, non-interactively,
# with this repository bind-mounted at /workspace (the working directory).
# Meant for batch jobs (see apptainer/submit.sh); the exit status is the
# command's own.
#
#     apptainer/run.sh [options] [--] command ...
#     apptainer/run.sh dtm-voxelization cases/rialba.toml speed
#     apptainer/run.sh -o /scratch/$USER/run1 -- dtm-voxelization cases/rialba.toml speed
#
# Options: the same as apptainer/shell.sh (-i, -o, -d, -b, -a), and so are the
# environment variables (APPTAINER_BIN, DTM_IMAGE, DTM_OUTPUT). The command
# starts at the first argument that is not an option, or after `--`.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

options=()
while [ $# -gt 0 ]; do
    case "$1" in
        -i|--image|-o|--output|-d|--data|-b|--bind|-a|--apptainer)
            options+=("$1" "$2"); shift 2 ;;
        -h|--help)
            sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --) shift; break ;;
        *)  break ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "no command given; for an interactive shell use apptainer/shell.sh" >&2
    exit 1
fi

echo "host:    $(hostname)"
echo "start:   $(date)"
echo "command: $*"

# shell.sh runs the command with `apptainer exec` when one follows `--`
status=0
source "${here}/shell.sh" ${options[@]+"${options[@]}"} -- "$@" || status=$?

echo "end:     $(date), exit status ${status}"
exit "${status}"
