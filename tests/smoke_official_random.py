"""CPU smoke test of src/lipsync/official_runner.py with RANDOM weights (no downloads, no GPU).

Proves the glue works end to end - upstream import, device handling, ref modes, CFG + audio
sensitivity, tensor dumps, paste-back, restore, full-rate audio mux - on a synthetic 1 s clip.
The output video is noise, of course; this checks plumbing, not quality.

  ULTRON_LATENTSYNC_DIR=/path/to/LatentSync  <venv with upstream deps>/bin/python tests/smoke_official_random.py
insightface is replaced by a fake detector with synthetic landmarks.
"""
import os
import subprocess
import sys
import tempfile
import types
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
UP = Path(os.environ["ULTRON_LATENTSYNC_DIR"]).resolve()


def fake_insightface():
    class Face:
        def __init__(self, k):
            lmk = np.zeros((106, 2), np.float32)
            lmk[:, 0] = np.linspace(70, 186, 106)
            lmk[:, 1] = 150
            lmk[0:33] = np.stack([np.linspace(70, 186, 33), 200 + 15 * np.sin(np.linspace(0, np.pi, 33))], 1)
            lmk[[43, 48, 49, 50, 51]] = [100, 95]
            lmk[101:106] = [156, 95]
            lmk[[73, 74, 77, 83, 86]] = [128, 135]
            o = 4 + 6 * (k % 5)  # mouth opening varies, frame 0,5,10.. most closed
            lmk[52:64] = np.stack([np.linspace(110, 146, 12), np.full(12, 170.0)], 1)
            lmk[64:68] = np.stack([np.linspace(114, 142, 4), np.full(4, 170 - o / 2)], 1)
            lmk[68:72] = np.stack([np.linspace(114, 142, 4), np.full(4, 170 + o / 2)], 1)
            self.landmark_2d_106 = lmk
            self.bbox = np.array([64, 60, 192, 224], np.float32)
            self.det_score = 0.99

    class FaceAnalysis:
        n = 0

        def __init__(self, *a, **k):
            pass

        def prepare(self, *a, **k):
            pass

        def get(self, img):
            FaceAnalysis.n += 1
            return [Face(FaceAnalysis.n - 1)]

    m = types.ModuleType("insightface")
    app = types.ModuleType("insightface.app")
    app.FaceAnalysis = FaceAnalysis
    m.app = app
    sys.modules["insightface"] = m
    sys.modules["insightface.app"] = app


def build_fake_upstream(tmp: Path):
    ls = tmp / "LatentSync"
    ls.mkdir()
    for d in ("configs", "latentsync"):
        (ls / d).symlink_to(UP / d)
    import torch

    sys.path.insert(0, str(ls))
    from latentsync.whisper.whisper.model import ModelDimensions, Whisper
    from latentsync.models.unet import UNet3DConditionModel
    from diffusers import AutoencoderKL
    from omegaconf import OmegaConf

    torch.manual_seed(0)
    dims = ModelDimensions(80, 1500, 384, 6, 4, 51865, 448, 384, 6, 4)
    (ls / "checkpoints" / "1.5" / "whisper").mkdir(parents=True)
    torch.save({"dims": asdict(dims), "model_state_dict": Whisper(dims).state_dict()}, ls / "checkpoints/1.5/whisper/tiny.pt")
    cfg = OmegaConf.load(str(UP / "configs/unet/stage2.yaml"))
    unet = UNet3DConditionModel.from_config(OmegaConf.to_container(cfg.model))
    with torch.no_grad():  # zero-initialised modules (audio cross-attn outputs, motion modules) -> random
        for k, v in unet.named_parameters():
            if v.abs().sum() == 0:
                v.normal_(0, 0.02)
    torch.save({"state_dict": {k: v.half() for k, v in unet.state_dict().items()}}, ls / "checkpoints/1.5/latentsync_unet.pt")
    del unet
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 32, 64, 64),
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        layers_per_block=1, norm_num_groups=32, sample_size=256)
    vae.save_pretrained(str(tmp / "vae"))
    return ls


def main():
    fake_insightface()
    tmp = Path(tempfile.mkdtemp(prefix="ls_smoke_"))
    ls = build_fake_upstream(tmp)
    os.environ["ULTRON_LATENTSYNC_VAE"] = str(tmp / "vae")
    vid, wav = tmp / "in video.mp4", tmp / "voice.wav"  # space on purpose
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=30:duration=1.2",
                    "-pix_fmt", "yuv420p", str(vid)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=1.0:sample_rate=44100",
                    str(wav)], check=True)
    from lipsync import official_runner

    results = {}
    for mode in ("same", "window", "fixed"):
        out = tmp / f"out_{mode}.mp4"
        args = Namespace(video=str(vid), audio=str(wav), out=str(out), version="1.5", guidance=2.0, steps=2, seed=1,
                         resolution=None, ref_mode=mode, ref_idx="auto" if mode == "fixed" else None, ref_shift=None,
                         device="cpu", dtype="fp32", audio_device="cpu", deepcache=False, latentsync_dir=str(ls),
                         ckpt=None, dump=str(tmp / f"dump_{mode}") if mode == "same" else None, dump_chunks=1,
                         parity=mode == "same", report=str(out) + ".json")
        rep = official_runner.run(args)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                                   capture_output=True, text=True).stdout)
        sr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=sample_rate", "-of", "csv=p=0",
                             str(out)], capture_output=True, text=True).stdout.strip()
        assert out.exists() and dur > 0.9, (mode, dur)
        assert sr == "44100", f"expected original-rate audio, got {sr}"
        assert rep["audio_sensitivity_mean"] is not None and rep["audio_sensitivity_mean"] > 0
        results[mode] = (rep["ref"]["first_indices"][:20], rep["audio_sensitivity_mean"], rep["frames"])
        print(mode, results[mode])
    assert results["same"][0][:16] == list(range(16))
    assert len(set(results["fixed"][0])) == 1
    w = results["window"][0]
    assert len(set(w)) == 1, w  # clip shorter than 2 windows -> falls back to fixed/auto (unit tests cover long clips)
    from lipsync import parity

    rows = parity.compare(tmp / "dump_same", tmp / "dump_same")
    assert all(r["status"] in ("ok", "info") for r in rows)
    names = {r["name"] for r in rows}
    assert any("step0/unet/mid_block" in n for n in names), names
    print("SMOKE OK", tmp)


if __name__ == "__main__":
    main()
