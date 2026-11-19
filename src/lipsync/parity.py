"""Find where the MLX LatentSync port stops matching the official PyTorch code.

Plan (all on the same inputs, one heavy model at a time):

 1. weights   - is every official tensor present in the MLX weights, with the same values?
                (upstream loads with strict=False, so a port can silently miss e.g. the audio
                cross-attention and still "run"; 1.6 vs 1.5 mix-ups show up here too)
       python src/lipsync/parity.py weights checkpoints/latentsync_unet.pt ~/…/latentsync-mlx/<weights>

 2. dump      - official run on a short clip with --parity --dump ref_dump/  (official_runner.py)
 3. teacher   - feed the official tensors from ref_dump/ into the MLX modules stage by stage
                (audio encoder, VAE encode, ONE UNet call, VAE decode) and save what MLX returns
       python src/lipsync/parity.py teacher ref_dump/ mlx_dump/ --adapter src/lipsync/mlx_adapter.py
 4. compare   - per tensor: shape, max |diff|, relative L2, cosine; first stage that diverges
       python src/lipsync/parity.py compare ref_dump/ mlx_dump/

Teacher forcing keeps an early small error from snowballing, so the first FAILED line is the bug.
The audio check needs no MLX internals: `teacher` also runs the UNet with zeroed audio and reports
audio_sensitivity = ||eps(audio) - eps(0)|| / ||eps(0)|| for both implementations.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ["audio/", "source/", "noise/", "faces", "audio_chunks", "pixel_values", "masked_pixel_values", "masks",
         "ref_pixel_values", "mask_latents", "masked_image_latents", "ref_latents", "latents_init",
         "step0/audio_embeds", "step0/unet_input", "step0/unet/conv_in", "step0/unet/down_blocks",
         "step0/unet/mid_block", "step0/unet/up_blocks", "step0/unet/conv_out", "step0/noise_pred",
         "latents_final", "decoded"]


def _rank(name: str) -> int:
    for i, k in enumerate(ORDER):
        if k in name:
            return i
    return len(ORDER)


def load_dump(d: Path) -> dict:
    return {str(p.relative_to(d))[:-4]: p for p in sorted(d.rglob("*.npy"))}


def align(a: np.ndarray, b: np.ndarray):
    """Return b permuted to a's layout if shapes are a permutation (NCHW vs NHWC etc.)."""
    if a.shape == b.shape:
        return b, None
    if a.size != b.size or a.ndim != b.ndim or a.ndim > 6:
        return None, None
    best = None
    for perm in itertools.permutations(range(b.ndim)):
        if tuple(b.shape[i] for i in perm) == a.shape:
            c = np.transpose(b, perm)
            e = float(np.linalg.norm((a - c).ravel()))
            if best is None or e < best[0]:
                best = (e, c, perm)
    return (best[1], best[2]) if best else (None, None)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    d = a - b
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {"max_abs": float(np.abs(d).max()) if d.size else 0.0, "rel_l2": float(np.linalg.norm(d) / (na + 1e-12)),
            "cos": float(a @ b / (na * nb + 1e-12)), "norm_ratio": float(nb / (na + 1e-12))}


def compare(ref_dir: Path, other_dir: Path, tol_rel=2e-2, tol_cos=0.999) -> list[dict]:
    A, B = load_dump(ref_dir), load_dump(other_dir)
    rows = []
    for name in sorted(set(A) & set(B), key=lambda n: (_rank(n), n)):
        a, b = np.load(A[name]), np.load(B[name])
        if a.dtype.kind not in "fiu":
            continue
        bb, perm = align(a, b)
        if bb is None:
            rows.append({"name": name, "status": "SHAPE", "ref": list(a.shape), "other": list(b.shape)})
            continue
        s = stats(a, bb)
        ok = s["rel_l2"] <= tol_rel or s["cos"] >= tol_cos
        rows.append({"name": name, "status": "ok" if ok else "FAILED", **{k: round(v, 6) for k, v in s.items()},
                     **({"perm": list(perm)} if perm else {})})
    only_ref = sorted(set(A) - set(B), key=_rank)
    if only_ref:
        rows.append({"name": f"(not in other: {len(only_ref)})", "status": "info", "sample": only_ref[:8]})
    return rows


# ------------------------------------------------------------------------------- weights audit
def load_weights(path: Path) -> dict:
    path = Path(path)
    files = sorted(path.rglob("*.safetensors")) + sorted(path.rglob("*.npz")) if path.is_dir() else [path]
    out = {}
    for f in files:
        if f.suffix == ".pt" or f.suffix == ".pth" or f.suffix == ".ckpt":
            import torch

            sd = torch.load(str(f), map_location="cpu", weights_only=True)
            sd = sd.get("state_dict", sd)
            out.update({k: v.float().numpy() for k, v in sd.items() if hasattr(v, "float")})
        elif f.suffix == ".npz":
            out.update({k: np.asarray(v, dtype=np.float32) for k, v in np.load(f).items()})
        elif f.suffix == ".safetensors":
            try:
                import mlx.core as mx

                for k, v in mx.load(str(f)).items():
                    out[k] = np.array(v.astype(mx.float32))
            except ImportError:
                from safetensors.numpy import load_file

                out.update({k: v.astype(np.float32) for k, v in load_file(str(f)).items()})
    return out


def _fp(v: np.ndarray):
    v = v.astype(np.float64).ravel()
    return (v.size, round(float(v.sum()), 3), round(float((v * v).sum()), 3), round(float(v.max()), 4), round(float(v.min()), 4))


def weights_audit(official: Path, other: Path, also: Path | None = None) -> dict:
    """Match tensors by value fingerprint (layout- and name-independent)."""
    O, M = load_weights(official), load_weights(other)
    idx = {}
    for k, v in M.items():
        idx.setdefault(_fp(v), []).append(k)
    by_numel = {}
    for k, v in M.items():
        by_numel.setdefault(v.size, []).append(k)
    missing, near, split, matched = [], [], [], 0
    for k, v in O.items():
        if _fp(v) in idx:
            matched += 1
            continue
        # split match: ports often store one fused tensor as equal chunks along dim 0
        # (GEGLU ff.net.0.proj -> linear1 + linear2 in the MLX port, fused qkv -> q, k, v)
        chunks_hit = None
        for n in (2, 3):
            if v.ndim and v.shape[0] % n == 0:
                parts = np.split(v, n, axis=0)
                if all(_fp(p) in idx for p in parts):
                    chunks_hit = [idx[_fp(p)][0] for p in parts]
                    break
        if chunks_hit:
            split.append((k, chunks_hit))
            continue
        # near match: same numel and close values in some layout (fp16/bf16 rounding in the port)
        hit = None
        vs, vq = float(v.sum()), float((v.astype(np.float64) ** 2).sum())
        for cand in by_numel.get(v.size, []):
            w = M[cand]
            ws, wq = float(w.sum()), float((w.astype(np.float64) ** 2).sum())
            if abs(wq - vq) > 1e-2 * (abs(vq) + 1e-6) or abs(ws - vs) > 1e-2 * (abs(vq) ** 0.5 + 1e-6) * 10:
                continue  # cheap order-free prefilter before trying layouts
            ww, _ = align(v, w) if w.shape != v.shape else (w, None)
            if ww is not None and stats(v, ww)["rel_l2"] < 5e-3:
                hit = cand
                break
        (near.append((k, hit)) if hit else missing.append(k))
    groups = {}
    for k in missing:
        g = ".".join(p for p in k.split(".") if not p.isdigit())
        groups[g] = groups.get(g, 0) + 1
    res = {"official_tensors": len(O), "other_tensors": len(M), "exact": matched, "near": len(near),
           "split_exact": len(split), "split_sample": [f"{k} = concat({', '.join(p)})" for k, p in split[:3]],
           "missing": len(missing), "missing_by_kind": dict(sorted(groups.items(), key=lambda t: -t[1])[:30]),
           "missing_sample": missing[:20],
           "audio_attention_missing": [k for k in missing if "attn2" in k][:20]}
    if also:  # e.g. does the "1.6" MLX folder actually hold 1.5 weights?
        A = load_weights(also)
        afp = {_fp(v) for v in A.values()}
        res["other_matches_also"] = sum(_fp(v) in afp for v in M.values())
    return res


# ------------------------------------------------------------------------------- teacher forcing
def teacher(ref_dir: Path, out_dir: Path, adapter_path: Path, chunk: int = 0) -> dict:
    """Run the adapter's modules on the official inputs and save outputs under the same names."""
    spec = importlib.util.spec_from_file_location("ls_adapter", adapter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ad = mod.Adapter()
    R = load_dump(ref_dir)
    man = json.loads((ref_dir / "manifest.json").read_text())
    rep = man.get("report", {})
    out_dir.mkdir(parents=True, exist_ok=True)

    def put(name, arr):
        p = out_dir / (name + ".npy")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.asarray(arr, dtype=np.float32))

    def get(name):
        return np.load(R[name])

    c = f"chunk{chunk}"
    summary = {}
    if hasattr(ad, "audio_chunks") and rep.get("audio"):
        put(f"{c}/audio_chunks", ad.audio_chunks(rep["audio"], start=chunk * 16, n=16))
    if hasattr(ad, "vae_encode"):
        for src, dst in (("masked_pixel_values", "masked_image_latents"), ("ref_pixel_values", "ref_latents")):
            # adapter contract: (f,3,H,W) in [-1,1] -> (f,4,H/8,W/8), latent MEAN times 0.18215
            z = np.asarray(ad.vae_encode(get(f"{c}/{src}")), dtype=np.float32)
            ref = get(f"{c}/{dst}")
            z = z.transpose(1, 0, 2, 3)[None]  # f c h w -> 1 c f h w (official layout)
            put(f"{c}/{dst}", np.concatenate([z] * ref.shape[0], 0))
    x, t, a = get(f"{c}/step0/unet_input"), get(f"{c}/step0/t"), get(f"{c}/step0/audio_embeds")
    eps = ad.unet(x, float(t), a, dump=lambda name, v: put(f"{c}/step0/unet/{name}", v))
    put(f"{c}/step0/noise_pred", eps)
    eps0 = ad.unet(x, float(t), np.zeros_like(a))
    summary["mlx_audio_sensitivity"] = float(np.linalg.norm(eps - eps0) / (np.linalg.norm(eps0) + 1e-12))
    ref_eps = get(f"{c}/step0/noise_pred")
    if ref_eps.shape[0] == 2:  # CFG batch: [uncond, cond]
        summary["official_audio_sensitivity_step0"] = float(np.linalg.norm(ref_eps[1] - ref_eps[0]) / (np.linalg.norm(ref_eps[0]) + 1e-12))
    if hasattr(ad, "vae_decode"):
        put(f"{c}/decoded", ad.vae_decode(get(f"{c}/latents_final")))
    (out_dir / "teacher_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def print_rows(rows):
    for r in rows:
        if r["status"] in ("ok", "FAILED"):
            print(f"{r['status']:6s} {r['name']:55s} rel_l2={r['rel_l2']:.2e} cos={r['cos']:.5f} "
                  f"max_abs={r['max_abs']:.3e} norm_ratio={r['norm_ratio']:.3f}" + (f" perm={r['perm']}" if "perm" in r else ""))
        else:
            print(f"{r['status']:6s} {r['name']} {json.dumps({k: v for k, v in r.items() if k not in ('name', 'status')})}")
    first = next((r for r in rows if r["status"] in ("FAILED", "SHAPE")), None)
    print("\nFIRST DIVERGENCE:", first["name"] if first else "none - implementations match on these tensors")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compare"); c.add_argument("ref"); c.add_argument("other"); c.add_argument("--json")
    c.add_argument("--tol-rel", type=float, default=2e-2)
    w = sp.add_parser("weights"); w.add_argument("official"); w.add_argument("other"); w.add_argument("--also")
    t = sp.add_parser("teacher"); t.add_argument("ref"); t.add_argument("out")
    t.add_argument("--adapter", default=str(Path(__file__).with_name("mlx_adapter.py"))); t.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "compare":
        rows = compare(Path(a.ref), Path(a.other), a.tol_rel)
        print_rows(rows)
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1))
    elif a.cmd == "weights":
        print(json.dumps(weights_audit(Path(a.official), Path(a.other), Path(a.also) if a.also else None), indent=1))
    else:
        print(json.dumps(teacher(Path(a.ref), Path(a.out), Path(a.adapter), a.chunk), indent=1))


if __name__ == "__main__":
    main()
