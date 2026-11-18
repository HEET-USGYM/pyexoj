#!/bin/bash
# Build ExoJCli against the installed ExoJ jar and Fiji's jars; install into $FIJI/plugins (default ~/Fiji).
set -euo pipefail
FIJI=${FIJI:-$HOME/Fiji}
JDK=$(ls -d "$FIJI"/java/linux64/*/ | head -1)
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(mktemp -d)
CP="$FIJI/plugins/ExoJ_1.9.2.jar:$(ls "$FIJI"/jars/ij-*.jar | head -1)"
"$JDK/bin/javac" --release 8 -nowarn -cp "$CP" -d "$B" "$HERE/ExoJCli.java"
"$JDK/bin/jar" cf "$FIJI/plugins/ExoJ_Cli.jar" -C "$B" .
rm -rf "$B"
echo "installed $FIJI/plugins/ExoJ_Cli.jar"
