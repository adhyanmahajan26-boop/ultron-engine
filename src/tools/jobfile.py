import json,sys
J='@STUDIO@/outputs/jobs'
d=json.load(open(f'{J}/{sys.argv[1]}.json')); print(d['result']['files'][0])
