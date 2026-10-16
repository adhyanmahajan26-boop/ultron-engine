"""Cloud GPU adapter (EXPERIMENTAL, not tested against a real provider by the authors).

Protocol (see docs/CLOUD.md): POST {url}/run  {"role": "...", "params": {...}}  -> {"job_id": "..."}
                              GET  {url}/jobs/{job_id}                         -> {"status": "queued|running|done|error", "result_url": "...", "error": "..."}
The API key is read from the environment variable NAMED in ultron.config.json (the key is never written to disk by this repo).
"""
import json, os, time, urllib.request
from pathlib import Path

CFG = json.loads((Path(__file__).resolve().parent.parent / "ultron.config.json").read_text())


def _req(url, key, data=None):
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.load(f)


def call(component_key, params, poll_s=5, timeout_s=3600):
    entry = next(v for k, v in CFG["cloud"].items() if k.endswith("." + component_key))
    key = os.environ[entry["api_key_env"]]
    job = _req(entry["url"].rstrip("/") + "/run", key, {"role": entry["adapter_role"], "params": params})["job_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = _req(entry["url"].rstrip("/") + f"/jobs/{job}", key)
        if st["status"] == "done":
            return st["result_url"]
        if st["status"] == "error":
            raise RuntimeError(st.get("error"))
        time.sleep(poll_s)
    raise TimeoutError(job)


if __name__ == "__main__":
    import sys
    print(call(sys.argv[1], json.loads(sys.argv[2])))
