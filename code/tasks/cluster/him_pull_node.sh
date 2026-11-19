#!/bin/bash
# node side of him_stage2.sh pulls: $1 = work dir, $2 = light | heavy | commit-light | commit-heavy.
# light: new files except resume state (marker .t0); heavy: new *.ckpt.pt and pregen batches (marker .h0); tar to stdout.
# commit-*: advance the marker (called by the login side only after its tar succeeded).
cd $1 || exit 1
case $2 in
  light) touch .t1; find revision_runs line*.log -newer .t0 -type f ! -path '*/phi4_buffer/*' ! -name '*.ckpt.pt' ! -path '*/pregen/*' 2>/dev/null | tar cf - -T - ;;
  heavy) touch .h1; find revision_runs -newer .h0 -type f \( -name '*.ckpt.pt' -o -path '*/pregen/*' \) 2>/dev/null | tar cf - -T - ;;
  commit-light) mv .t1 .t0 ;;
  commit-heavy) mv .h1 .h0 ;;
esac
