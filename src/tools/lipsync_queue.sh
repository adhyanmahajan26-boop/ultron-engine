#!/bin/sh
cd @ULTRON_ROOT@
PY=@PYTHON@
S=@AVATAR@/scripts/lipsync.py
for k in L1 L4 L9 L5 L7 L6 L8; do
  [ -f ls/$k.mp4 ] && continue
  echo "$(date +%H:%M:%S) start $k" >> logs/ls_queue.log
  $PY $S --version 1.6 --video src/$k.mp4 --audio audio/$k.wav --out ls/$k.mp4 > logs/ls_$k.log 2>&1 \
    && echo "$(date +%H:%M:%S) ok $k" >> logs/ls_queue.log \
    || echo "$(date +%H:%M:%S) FAIL $k" >> logs/ls_queue.log
done
echo QUEUEDONE >> logs/ls_queue.log
