"""TEMPLATE for the server you run on a rented GPU (RunPod, Modal, Lambda, your own box...). It implements the protocol in docs/CLOUD.md
but contains NO model code: fill in `run_job` with your CUDA build of the model for each `role` (for example upstream Lightricks LTX-2.5 for video.ltx).
Run:  pip install fastapi uvicorn ; ULTRON_WORKER_KEY=choose-a-secret uvicorn worker_template:app --host 0.0.0.0 --port 8000
"""
import os, threading, uuid
from fastapi import FastAPI, Header, HTTPException

app, JOBS = FastAPI(), {}


def auth(h):
    if h != f"Bearer {os.environ['ULTRON_WORKER_KEY']}":
        raise HTTPException(401)


def run_job(job_id, role, params):
    try:
        raise NotImplementedError(f"plug your model for role {role!r} in here; write the output file somewhere you can serve and set result_url")
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/run")
def run(body: dict, authorization: str = Header("")):
    auth(authorization)
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"status": "running"}
    threading.Thread(target=run_job, args=(jid, body["role"], body["params"]), daemon=True).start()
    return {"job_id": jid}


@app.get("/jobs/{jid}")
def job(jid: str, authorization: str = Header("")):
    auth(authorization)
    return JOBS[jid]
