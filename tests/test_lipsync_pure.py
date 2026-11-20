"""Pure-python tests for src/lipsync (numpy only): python -m pytest tests/test_lipsync_pure.py  (or run directly)."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lipsync import backends, config, parity, refmodes  # noqa: E402


def test_same_is_identity():
    assert refmodes.ref_indices("same", 40) == list(range(40))


def test_fixed_auto_picks_most_closed_usable():
    o = np.array([0.05, 0.01, 0.03, 0.002, 0.04])
    assert refmodes.ref_indices("fixed", 5, idx="auto", openness=o) == [3] * 5
    assert refmodes.ref_indices("fixed", 5, idx="auto", openness=o, usable=[1, 1, 1, 0, 1]) == [1] * 5
    assert refmodes.ref_indices("fixed", 5, idx=2) == [2] * 5


def test_shift_wraps():
    assert refmodes.ref_indices("shift", 20, shift=16)[:6] == [16, 17, 18, 19, 0, 1]


def test_window_is_training_like():
    n = 250  # a 10 s take
    rng = np.random.default_rng(0)
    o = rng.uniform(0.02, 0.08, n)
    o[100:116] = 0.001  # the calm, closed-mouth stretch
    r = refmodes.ref_indices("window", n, openness=o)
    assert len(r) == n
    for s in range(0, n, 16):
        span = min(16, n - s)
        start = r[s]
        assert r[s:s + span] == list(range(start, start + span))  # contiguous window, same motion order
        assert start + 16 <= s or start >= s + 16  # never overlaps the chunk being redrawn (as in training)
    assert r[0] == 100 and r[192] == 100 and r[200] == 108  # picks the closed-mouth window when allowed
    assert r[100] != 100


def test_window_short_clip_falls_back():
    assert len(set(refmodes.ref_indices("window", 20, openness=np.linspace(0.1, 0.0, 20)))) == 1


def test_env_compat():
    assert refmodes.from_env({"LS_REF_MODE": "fixed", "LS_REF_IDX": "86"}) == {"mode": "fixed", "idx": 86, "shift": None}


def test_backend_commands():
    j = backends.Job(video="a b.mp4", audio="v.wav", out="o.mp4", backend="official", guidance=2.0, ref_mode="window")
    cmd, env, cwd = backends.command(j)
    assert cmd[1].endswith("official_runner.py") and "--ref-mode" in cmd and "window" in cmd and "a b.mp4" in cmd
    j = backends.Job(video="a.mp4", audio="v.wav", out="o.mp4", backend="mlx", ref_mode="window")
    cmd, env, cwd = backends.command(j)
    assert env["LS_REF_MODE"] == "shift" and env["LS_REF_SHIFT"] == "16" and cmd[1] == "scripts/lipsync.py"


def test_checkpoint_layouts():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "checkpoints").mkdir()
        (d / "checkpoints" / "latentsync_unet.pt").write_bytes(b"x")
        assert config.checkpoint("1.6", d) == d / "checkpoints" / "latentsync_unet.pt"
        assert not config.checkpoint("1.5", d).exists()  # flat file is never taken for 1.5
        (d / "checkpoints" / "1.6").mkdir()
        (d / "checkpoints" / "1.6" / "latentsync_unet.pt").write_bytes(b"y")
        assert config.checkpoint("1.6", d) == d / "checkpoints" / "1.6" / "latentsync_unet.pt"


def test_parity_compare_handles_nhwc_and_flags_divergence():
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        rng = np.random.default_rng(1)
        x = rng.normal(size=(1, 4, 16, 8, 8)).astype(np.float32)
        y = rng.normal(size=(2, 4, 16, 8, 8)).astype(np.float32)
        for root, xx, yy in ((a, x, y), (b, np.transpose(x, (0, 2, 3, 4, 1)), y * 0.5 + 0.3)):
            (root / "chunk0" / "step0").mkdir(parents=True)
            np.save(root / "chunk0" / "latents_init.npy", xx)
            np.save(root / "chunk0" / "step0" / "noise_pred.npy", yy)
        rows = {r["name"]: r for r in parity.compare(a, b)}
        assert rows["chunk0/latents_init"]["status"] == "ok" and rows["chunk0/latents_init"]["perm"]
        assert rows["chunk0/step0/noise_pred"]["status"] == "FAILED"


def test_qa_lag_corr_finds_delay():
    import importlib.util
    spec = importlib.util.spec_from_file_location("qa", Path(__file__).resolve().parents[1] / "src/lipsync/qa.py")
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)  # insightface is only imported lazily
    rng = np.random.default_rng(3)
    voice = np.convolve(rng.random(300), np.ones(5) / 5, "same")
    mouth = np.roll(voice, 2)  # mouth 2 frames (80 ms) after the voice
    lag, best, c0 = qa.lag_corr(voice, mouth)
    assert lag == 2 and best > 0.95 and c0 < best
    assert qa.verdict({"voice_corr_best": 0.6, "best_lag_ms": 0, "identity": {"mean": 0.9},
                       "silent": {"speech_vs_silent_corr": 0.95, "silent_open_ratio": 0.2}})[0] == "FAIL"




def test_nearest5d_as_2d_matches_3d():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return
    from lipsync import official_runner
    x = torch.randn(2, 5, 16, 8, 12)
    want = F.interpolate(x, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
    got = official_runner.nearest5d_as_2d(F.interpolate, x, scale_factor=[1.0, 2.0, 2.0])
    assert torch.equal(want, got)
    want = F.interpolate(x, size=(16, 24, 36), mode="nearest")
    assert torch.equal(want, official_runner.nearest5d_as_2d(F.interpolate, x, size=(16, 24, 36)))
    assert official_runner.nearest5d_as_2d(F.interpolate, x, scale_factor=2.0) is None  # time axis scaled
    assert official_runner.nearest5d_as_2d(F.interpolate, x, scale_factor=[1, 2, 2], mode="trilinear") is None

def test_sliced_sdpa_matches_full():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return
    from lipsync import official_runner
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 4, 40, 16), torch.randn(3, 4, 40, 16), torch.randn(3, 4, 40, 16)
    want = F.scaled_dot_product_attention(q, k, v)
    for budget in (40 * 40 * 4 * 3 * 5, 40 * 4 * 3 * 7, 1):   # whole (b*h) slices, query-row slices, one row at a time
        got = official_runner.sliced_sdpa(F.scaled_dot_product_attention, q, k, v, budget=budget)
        assert torch.allclose(want, got, atol=1e-6), budget
    assert official_runner.sliced_sdpa(F.scaled_dot_product_attention, q, k, v, budget=2**30) is None  # fits whole
    assert official_runner.sliced_sdpa(F.scaled_dot_product_attention, q, k, v, attn_mask=torch.ones(40, 40), budget=1) is None


def test_single_instance_lock():
    import subprocess
    from lipsync import official_runner
    with tempfile.TemporaryDirectory() as d:
        os.environ["ULTRON_LOCK_DIR"] = d
        try:
            official_runner.single_instance("t")
            code = ("import sys; sys.path.insert(0, %r); from lipsync import official_runner as o; o.single_instance('t')"
                    % str(Path(__file__).resolve().parents[1] / "src"))
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ)
            assert r.returncode != 0 and "another LatentSync run" in r.stderr, r.stderr
        finally:
            official_runner._LOCK_FH.close()
            official_runner._LOCK_FH = None
            del os.environ["ULTRON_LOCK_DIR"]


def test_mouthblend_keeps_outside_and_fixes_pink():
    try:
        import cv2  # noqa: F401
    except ImportError:
        return
    from lipsync import mouthblend
    h, w = 200, 200
    src = np.full((h, w, 3), (60, 80, 110), np.uint8)       # skin/beard-ish, BGR
    gen = src.copy()
    gen[60:200, :] = (40, 40, 200)                           # pink/red smear over the whole lower half
    lm = np.zeros((106, 2), np.float32)
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    lm[52:72] = np.stack([100 + 30 * np.cos(ang), 120 + 12 * np.sin(ang)], 1)
    out = mouthblend.blend([src], [gen], [lm], [lm], margin=0.2, feather=0.05, chroma=1.0)[0]
    assert np.abs(out[10, 10].astype(int) - src[10, 10]).max() <= 1       # far from the mouth: source
    assert np.abs(out[190, 10].astype(int) - src[190, 10]).max() <= 1     # beard below/aside: source, not smear
    red_gen = int(gen[120, 100, 2]) - int(gen[120, 100, 0])
    red_out = int(out[120, 100, 2]) - int(out[120, 100, 0])
    assert red_out < red_gen                                               # pink pulled toward source colour


def test_mouthblend_covers_wider_source_mouth():
    try:
        import cv2  # noqa: F401
    except ImportError:
        return
    from lipsync import mouthblend
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    small, wide = np.zeros((106, 2), np.float32), np.zeros((106, 2), np.float32)
    small[52:72] = np.stack([100 + 25 * np.cos(ang), 100 + 6 * np.sin(ang)], 1)
    wide[52:72] = np.stack([100 + 30 * np.cos(ang), 100 + 22 * np.sin(ang)], 1)
    m = mouthblend.mouth_mask((200, 200, 3), small, wide, margin=0.12, feather=0.10)
    inside = np.zeros((200, 200), np.uint8)
    cv2.fillConvexPoly(inside, cv2.convexHull(wide[52:72].astype(np.int32)), 1)
    assert m[inside > 0].min() > 0.97          # the whole source mouth is replaced, no ghost


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
