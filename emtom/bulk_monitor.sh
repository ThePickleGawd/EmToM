#!/bin/bash
# Monitor EMTOM bulk generation progress
#
# Usage:
#   ./emtom/bulk_monitor.sh                              # Latest run, refresh every 5s
#   ./emtom/bulk_monitor.sh outputs/emtom/<dir>          # Specific run
#   ./emtom/bulk_monitor.sh --interval 2                 # Faster refresh
#   ./emtom/bulk_monitor.sh --once                       # Print once, no refresh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/bulk_monitor.py" "$@"
