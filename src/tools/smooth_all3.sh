#!/bin/zsh
cd @ULTRON_ROOT@
PY=@PYTHON@
for f in ls3/*.mp4; do n=${f:t:r}; [ -f ls3s/$n.mp4 ] && continue
  $PY tools/smooth_mouth.py ls3/$n.mp4 audio/slices3/$n.wav ls3s/$n.mp4 0.6 2>&1 | tail -1; done
