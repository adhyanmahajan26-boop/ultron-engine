#!/bin/zsh
cd @ULTRON_ROOT@
PY=@PYTHON@
for f in ls2/*.mp4; do n=${f:t:r}
  [ -f ls2s/$n.mp4 ] && continue
  $PY tools/smooth_mouth.py ls2/$n.mp4 audio/slices/$n.wav ls2s/$n.mp4 0.6 2>&1 | tail -1
done
