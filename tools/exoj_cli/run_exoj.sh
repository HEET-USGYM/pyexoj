#!/bin/bash
# Usage: run_exoj.sh settings.properties [timeout_s]
# Runs ExoJ (via ExoJCli) headless under xvfb in $FIJI (default ~/Fiji). Needs xvfb-run.
set -u
FIJI=${FIJI:-$HOME/Fiji}
HERE=$(cd "$(dirname "$0")" && pwd)
rm -f /tmp/ImageJ-$USER-*.stub
timeout "${2:-7200}" xvfb-run -a "$FIJI/fiji" -batch "$HERE/exoj_run.ijm" "$1"
rc=$?
rm -f /tmp/ImageJ-$USER-*.stub
exit $rc
