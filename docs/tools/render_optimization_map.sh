#!/usr/bin/env bash
# Render docs/AMD_OPTIMIZATION_MAP.md into a self-contained, theme-aware HTML page
# (the same page published as the Claude Code artifact). Requires pandoc >= 3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-$HERE/../AMD_OPTIMIZATION_MAP.md}"
OUT="${2:-/tmp/arcto_amd_optimization_map.html}"
pandoc -f gfm+pipe_tables -t html5 --section-divs --wrap=none --toc --toc-depth=2 \
  --shift-heading-level-by=-1 --metadata title="ARCTO AMD Optimization Map" \
  --template="$HERE/optimization_map_template.html" "$SRC" > "$OUT"
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
