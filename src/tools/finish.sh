#!/bin/zsh
# usage: finish.sh VERSION   -> renders film/ to review/film_VERSION.mp4, lifts loudness to -14 LUFS, writes phone copy
cd @ULTRON_ROOT@
export PATH=@PNPM@:$PATH
FF=@FFMPEG@; v=$1
cd film && npx hyperframes render -o ../review/film_${v}_raw.mp4 -q looks -f 30 > ../logs/render_${v}.log 2>&1; cd ..
$FF -hide_banner -loglevel error -y -i review/film_${v}_raw.mp4 -c:v copy -af "loudnorm=I=-14:TP=-1.5:LRA=9" -ar 48000 -c:a aac -b:a 224k review/film_${v}.mp4
$FF -hide_banner -loglevel error -y -i review/film_${v}.mp4 -vf scale=1280:-2 -c:v libx264 -crf 25 -preset medium -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart review/film_${v}_phone.mp4
$FF -hide_banner -i review/film_${v}.mp4 -af ebur128=peak=true -f null - 2>&1 | grep -E "I:|Peak:" | tail -2
ls -la review/film_${v}*.mp4 | awk '{print $5,$9}'
