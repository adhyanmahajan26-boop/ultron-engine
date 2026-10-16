#!/bin/zsh
# usage: heavy_guard.sh [max_used_gb=55]   -> exit 0 only if NO other heavy job is running and memory is safe. Run before every heavy job:  tools/heavy_guard.sh && <job>
max=${1:-55}
busy=$(pgrep -fl "mflux-generate|ltx-2-mlx (generate|a2v)|scripts/lipsync.py|scripts/inference.py|hyperframes render|mlx_lm|llama" | grep -v pgrep | cut -c1-70)
if [ -n "$busy" ]; then echo "GUARD: another heavy job is running -> NOT starting:"; echo "$busy"; exit 1; fi
loaded=$(curl -s -m 3 http://127.0.0.1:11234/v1/models 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print(sum(m.get('bytes_resident',0) for m in d)/1e9)" 2>/dev/null)
if [ -n "$loaded" ] && [ "$(python3 -c "print(1 if float('$loaded')>1 else 0)")" = "1" ]; then echo "GUARD: studio server holds ${loaded} GB of models -> call free_memory first"; exit 1; fi
used=$(vm_stat | awk '/page size/{ps=$8} /Pages active/{a=$3} /Pages wired/{w=$4} /occupied by compressor/{c=$5} END{printf "%d",(a+w+c)*ps/1e9}')
if [ "$used" -gt "$max" ]; then echo "GUARD: memory used ${used} GB > ${max} GB -> NOT starting"; exit 1; fi
echo "GUARD ok: no heavy job running, studio models unloaded, memory used ${used} GB"; exit 0
