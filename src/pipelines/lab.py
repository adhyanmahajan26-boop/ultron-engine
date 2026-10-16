import subprocess, sys, time
P="@ULTRON_ROOT@/priest"
STYLE=" Live-action cinematic film, 35mm grain, teal and crimson grading."
LAB={
 "F1":{"old":"The priest completes a fast silver sword slash, the leaping vampire's body splits apart and bursts into glowing red embers, ash and black blood spraying in slow motion, rain falling, the priest's coat whipping, camera slowly orbiting him, dramatic red rim light.",
       "new":"The priest on the left follows through his rightward sword swing, the blade still arcing as the pale vampire on the right, headless, stiffens and staggers one step forward while its head dissolves into a rolling cloud of grey ash and glowing red embers. Black blood sprays in a thin arc across the rain, then the vampire's knees buckle and the body collapses onto the wet stone. The priest steps back and flicks the blade down at his side, his long coat swinging. The handheld camera pushes in slowly on the priest, rain streaking diagonally through red light."},
 "S06":{"old":"The priest completes a fast silver sword slash, the leaping vampire's body splits apart and bursts into glowing red embers, ash and black blood spraying in slow motion, rain falling, the priest's coat whipping, camera slowly orbiting him, dramatic red rim light.",
       "new":"On the left the priest whips his silver sword across in a fast rightward slash, and the leaping vampire on the right is cut through the chest mid-air. The vampire's body splits and bursts outward into glowing red embers and ash that blow past the camera, then the priest strides forward through the cloud, his black coat snapping behind him. He lowers the blade and looks down at the drifting ash. The camera tracks sideways with the swing, handheld, then slows to a gentle push toward his face while rain falls through red light."},
 "F10":{"old":"The priest and Luka clash swords with sparks, Luka's claws slash the priest's coat, the priest headbutts Luka, stained glass glowing, cinematic gritty look.",
       "new":"In the foreground the priest, seen from behind, raises his silver sword in both hands as Luka lunges from the right, and the two blades clash at the center of the frame, a burst of orange sparks flying toward the camera. Luka forces forward with a second slash, the priest turns his shoulder, ducks under it and shoves Luka backward with his boot, Luka skidding across the stone floor. Colored stained-glass light flickers across their coats and faces. The handheld camera pushes in on the clash, then drifts left around the priest."},
}
for sid in LAB:
    for kind in ("old","new"):
        out=f"{P}/shots/LAB_{sid}_{kind}.mp4"
        cmd=f'cd @LTX@/ltx-2-mlx-xoc && while ! @ULTRON_ROOT@/tools/heavy_guard.sh >/dev/null; do sleep 15; done; s=$(date +%s); uv run ltx-2-mlx generate --distilled --model @LTX@/pack --prompt "{LAB[sid][kind]+STYLE}" --image {P}/stills/{sid}_1024.png --height 576 --width 1024 --frames 97 --frame-rate 24 --seed 701 -o {out} && echo "LAB_{sid}_{kind} done $(( $(date +%s)-s ))s" >> {P}/logs/status.txt'
        subprocess.run(["zsh","-c",cmd])
