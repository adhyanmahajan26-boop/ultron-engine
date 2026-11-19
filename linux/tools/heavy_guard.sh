#!/usr/bin/env bash
# Linux version of heavy_guard.sh (same contract as the Mac one):
#   heavy_guard.sh [max_used_ram_gb=auto] [max_used_vram_gb=2]  -> exit 0 only if NO other heavy job runs and memory is safe.
# Run before every heavy job:  heavy_guard.sh && <job>
max_ram=${1:-auto}
max_vram=${2:-2}
busy=$(pgrep -fa "scripts/lipsync.py|scripts/inference.py|src.lipsync.run|text2semantic/inference.py|hyperframes render|ltx_pipelines|mflux-generate" | grep -v -e pgrep -e heavy_guard | cut -c1-90)
if [ -n "$busy" ]; then echo "GUARD: another heavy job is running -> NOT starting:"; echo "$busy"; exit 1; fi
port=${CUDA_SERVE_PORT:-11234}
loaded=$(curl -s -m 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print(sum(m.get('bytes_resident',0) for m in d)/1e9)" 2>/dev/null)
if [ -n "$loaded" ] && python3 -c "import sys; sys.exit(0 if float('$loaded')>1 else 1)"; then
  echo "GUARD: cuda-serve holds ${loaded} GB of models -> call free_memory (or: curl -X POST localhost:$port/v1/unload-model) first"; exit 1
fi
if command -v nvidia-smi >/dev/null; then
  vused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$vused" ] && [ "$((vused / 1024))" -gt "$max_vram" ]; then
    echo "GUARD: GPU memory in use ${vused} MiB (> ${max_vram} GB) -> NOT starting:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; exit 1
  fi
fi
total=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
used=$((total - avail))
[ "$max_ram" = auto ] && max_ram=$((total * 60 / 100))
if [ "$used" -gt "$max_ram" ]; then echo "GUARD: RAM used ${used} GB > ${max_ram} GB -> NOT starting"; exit 1; fi
echo "GUARD ok: no heavy job running, models unloaded, RAM used ${used}/${total} GB, VRAM used ${vused:-?} MiB"; exit 0
