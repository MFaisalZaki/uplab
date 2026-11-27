#!/usr/bin/env bash
# Sets up up-lab in a virtual environment next to this script.
#
#   ./setup.sh                 create .venv and install up-lab
#   ./setup.sh --example       ... then run the example experiment
#   ./setup.sh --python python3.12 --venv /path/to/venv
#
# Uses uv when it is installed, the standard venv and pip otherwise.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python=""
venv="$here/.venv"
example=false

while [ $# -gt 0 ]; do
    case "$1" in
        --python)  python="$2"; shift 2 ;;
        --venv)    venv="$2"; shift 2 ;;
        --example) example=true; shift ;;
        -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

supported() {  # Python >= 3.10
    "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null
}

if [ -z "$python" ]; then
    for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null && supported "$candidate"; then
            python="$candidate"
            break
        fi
    done
fi
if [ -z "$python" ] || ! supported "$python"; then
    echo "up-lab needs Python 3.10 or newer; give one with --python <interpreter>" >&2
    exit 1
fi
command -v git >/dev/null || echo "warning: git not found; benchmark sources given as git URLs will fail" >&2

packages=(-e "$here/packages/up-lab-worker" -e "$here/packages/up-lab[notebook,serve]" -e "$here/packages/up-lab-stats")

echo "Python:      $("$python" --version 2>&1) ($(command -v "$python"))"
echo "Environment: $venv"
if command -v uv >/dev/null; then
    echo "Installer:   uv"
    uv venv --quiet --allow-existing --python "$(command -v "$python")" "$venv"
    uv pip install --quiet --python "$venv/bin/python" "${packages[@]}"
else
    echo "Installer:   venv + pip (uv not found)"
    "$python" -m venv "$venv"
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet "${packages[@]}"
fi
"$venv/bin/uplab" --help >/dev/null
echo "up-lab is installed."

if $example; then
    echo "Running the example experiment ..."
    "$venv/bin/uplab" all "$here/examples/experiment.yaml"
    echo "Results: $here/examples/experiments/demo/reports"
fi

cat <<MSG

Next:
  source "$venv/bin/activate"
  uplab all "$here/examples/experiment.yaml"     # local example
  uplab --help
MSG
