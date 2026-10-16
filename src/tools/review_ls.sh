#!/bin/zsh
# usage: review_ls.sh NAME  -> prints face sim, writes scratch/rv_NAME.png (full-frame + mouth strips)
cd @ULTRON_ROOT@
FF=@FFMPEG@; PY=@PYTHON@
n=$1
$FF -hide_banner -i ls2/$n.mp4 2>&1 | grep -E "Duration|Video:" | cut -c1-110
$PY tools/face_sim.py ls2/$n.mp4 6 stills/s_48.png stills/s_112.png stills/s_56.png scratch/ref_24.png stills/armor_face_f46.png stills/armor_nyc_1.png stills/shirt_cobalt.png stills/shirt_crimson.png stills/shirt_white.png 2>&1 | tail -1
$FF -hide_banner -loglevel error -y -i ls2/$n.mp4 -vf "select='not(mod(n,5))',scale=360:-1,tile=6x1" -frames:v 1 scratch/rv_$n.png
