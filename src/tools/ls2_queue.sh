#!/bin/sh
cd @ULTRON_ROOT@
PY=@PYTHON@
S=@AVATAR@/scripts/lipsync.py
for k in W1a W1b W4a W4b W4c W4d W5a W5b W5c W5d W5e W6a W6b W7a W7b W8 W9a spread_b speak_nyc; do
  [ -f ls2/$k.mp4 ] && continue
  echo "$(date +%H:%M:%S) start $k" >> logs/ls2_queue.log
  $PY $S --version 1.6 --video clips/$k.mp4 --audio audio/slices/$k.wav --out ls2/$k.mp4 > logs/ls2_$k.log 2>&1 \
    && echo "$(date +%H:%M:%S) ok $k" >> logs/ls2_queue.log \
    || echo "$(date +%H:%M:%S) FAIL $k" >> logs/ls2_queue.log
done
echo QUEUEDONE >> logs/ls2_queue.log
