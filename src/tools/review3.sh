#!/bin/zsh
cd @ULTRON_ROOT@
FF=@FFMPEG@; PY=@PYTHON@
n=$1
$PY tools/face_sim.py ls3s/$n.mp4 5 stills/s_48.png stills/s_112.png stills/s_56.png scratch/ref_24.png stills/shirt_cobalt.png stills/shirt_crimson.png stills/shirt_white.png 2>&1 | tail -1
$FF -hide_banner -i ls3s/$n.mp4 2>&1 | grep Duration
$FF -hide_banner -loglevel error -y -i ls3s/$n.mp4 -vf "select='not(mod(n,3))',crop=330:200:260:190,scale=240:-1,tile=8x1" -frames:v 1 scratch/rv3_$n.png
