#!/bin/zsh
# usage: ramlog.sh out.csv  -> every 5 s: epoch, used_GB (active+wired+compressed), wired_GB, swap_used_MB. kill to stop.
out=$1; echo "epoch,used_gb,wired_gb,swap_mb" > $out
while true; do
  vm_stat | awk -v t=$(date +%s) -v sw="$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d 'M')" '/page size/{ps=$8} /Pages active/{a=$3} /Pages wired/{w=$4} /occupied by compressor/{c=$5} END{printf "%d,%.1f,%.1f,%s\n",t,(a+w+c)*ps/1e9,w*ps/1e9,sw}' >> $out
  sleep 5
done
