#!/usr/bin/env bash
# Check the Hailo setup on this Pi with the system Python, where apt installs hailo_platform.
# Works from a plain clone: no pip install needed. Arguments go to the doctor (e.g. --help).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" exec /usr/bin/env python3 -m hailo_yolo.doctor "$@"
