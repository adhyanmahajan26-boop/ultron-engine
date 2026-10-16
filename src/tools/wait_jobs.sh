#!/bin/sh
# usage: wait_jobs.sh job1 job2 ... ; exits when all are done/error/cancelled
J=@STUDIO@/outputs/jobs
for j in "$@"; do until grep -qE '"status": "(done|error|cancelled)"' $J/$j.json 2>/dev/null; do sleep 15; done; done
for j in "$@"; do python3 -c "
import json;d=json.load(open('$J/$j.json'));r=d.get('result') or {};print('$j',d['status'],(r.get('files') or [d.get('error')])[0] if isinstance(r,dict) else r)"; done
