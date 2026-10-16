#!/bin/zsh
cd @ULTRON_ROOT@
PY=@PYTHON@
S=@AVATAR@/scripts/lipsync.py
J=@STUDIO@/outputs/jobs
idle() { $PY - <<'PYEOF'
import json,glob,sys
for f in glob.glob('@STUDIO@/outputs/jobs/*.json'):
    try:
        d=json.load(open(f))
        if d.get('status')=='running' and d.get('started','')>='2026-10-08T16:00:00' and d.get('kind') in ('video','image','image_edit','speech','music'): sys.exit(1)
    except Exception: pass
sys.exit(0)
PYEOF
}
# window:source-clip  (S1 and W9b are produced by H3 jobs still in flight: they go last)
LIST=(W4a:clips/W4a.mp4 W4b:clips/W4b.mp4 W4c:clips/W4c.mp4 S2:clips/W5b.mp4 S3:clips/W5a.mp4 S4:clips/W5d.mp4 S5:clips/W5c.mp4 W6a:clips/W6a.mp4 W6b:clips/W6b.mp4 W9a:clips/W9a.mp4 S1:clips/white_talk.mp4 W9b:clips/spread_new.mp4)
for item in $LIST; do
  n=${item%%:*}; src=${item#*:}
  [ -f ls3/$n.mp4 ] && continue
  until [ -f $src ] && idle; do sleep 10; done
  echo "$(date +%H:%M:%S) start $n" >> logs/ls3_queue.log
  $PY $S --version 1.6 --video $src --audio audio/slices3/$n.wav --out ls3/$n.mp4 > logs/ls3_$n.log 2>&1 \
    && echo "$(date +%H:%M:%S) ok $n" >> logs/ls3_queue.log || echo "$(date +%H:%M:%S) FAIL $n" >> logs/ls3_queue.log
done
echo QUEUEDONE >> logs/ls3_queue.log
