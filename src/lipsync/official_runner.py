"""Run ByteDance's official LatentSync (PyTorch) in-process, on CUDA, Apple MPS or CPU.

This is upstream's scripts/inference.py + LipsyncPipeline.__call__ (commit pinned in
src/lipsync/__init__.py) with only these changes, each one switchable so the result can be
compared against the stock pipeline:

  * device portability: no hard-coded "cuda" (face detector, aligner, whisper, pipeline)
  * reference-frame modes (--ref-mode same|fixed|shift|window, default "same" = upstream)
  * paste-back uses the TARGET frame (upstream uses the ref, which is the same thing in "same")
  * the output gets the ORIGINAL full-rate audio, not LatentSync's 16 kHz resample
  * paths with spaces work (no shell=True ffmpeg)
  * a report.json with the checkpoint key check, timings and the per-chunk "audio sensitivity"
    ||eps(audio) - eps(silence)|| / ||eps(silence)|| — ~0 means the UNet ignores the voice
  * --dump DIR writes named intermediate tensors for src/lipsync/parity.py; --parity makes the
    run deterministic (VAE mean instead of sample, DeepCache off, CPU-seeded noise saved to disk)

Run from anywhere with the LatentSync venv's python:
  $ULTRON_LATENTSYNC_PY -m lipsync.official_runner --video V --audio A --out O [--version 1.6]
(with src/ on PYTHONPATH, or via `python src/lipsync/run.py --backend official ...`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # svd etc. fall back to CPU instead of crashing

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipsync import UPSTREAM_COMMIT, config, refmodes  # noqa: E402

AUDIO_KEY_HINTS = ("attn2", "audio")


# ----------------------------------------------------------------------------- environment
def pick_device(want: str = "auto") -> str:
    import torch

    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, want: str = "auto"):
    import torch

    if want == "fp16":
        return torch.float16
    if want == "fp32":
        return torch.float32
    # upstream rule: fp16 only on CUDA with compute capability > 7; everything else fp32
    if device.startswith("cuda") and torch.cuda.get_device_capability()[0] > 7:
        return torch.float16
    return torch.float32


def import_upstream(ls_dir: Path):
    """Put the upstream checkout on sys.path and cd into it (upstream uses relative paths:
    configs/, latentsync/utils/mask.png, checkpoints/auxiliary)."""
    if not (ls_dir / "latentsync" / "pipelines" / "lipsync_pipeline.py").exists():
        raise SystemExit(f"LatentSync checkout not found at {ls_dir} (set ULTRON_LATENTSYNC_DIR or run "
                         f"scripts/setup_latentsync_official.sh)")
    head = ls_dir / ".git" / "HEAD"
    try:
        rev = subprocess.run(["git", "-C", str(ls_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = head.read_text().strip() if head.exists() else "?"
    if rev and rev != UPSTREAM_COMMIT:
        print(f"[warn] LatentSync checkout is at {rev[:10]}, tested commit is {UPSTREAM_COMMIT[:10]}", file=sys.stderr)
    sys.path.insert(0, str(ls_dir))
    os.chdir(ls_dir)
    return rev


# ----------------------------------------------------------------------------- classes built on upstream
def make_classes():
    import numpy as np
    import torch
    import torchvision
    from einops import rearrange
    from torchvision import transforms
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.utils.affine_transform import AlignRestore
    from latentsync.utils.face_detector import FaceDetector, INSIGHTFACE_DETECT_SIZE
    from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask

    class PortableFaceDetector(FaceDetector):
        def __init__(self, device="cuda"):  # noqa: D401 - same signature as upstream
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            avail = ort.get_available_providers()
            if device.startswith("cuda"):
                want, ctx = ["CUDAExecutionProvider", "CPUExecutionProvider"], (torch.device(device).index or 0)
            else:
                want, ctx = ["CoreMLExecutionProvider", "CPUExecutionProvider"], -1
            providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
            self.app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"], root="checkpoints/auxiliary",
                                    providers=providers)
            self.app.prepare(ctx_id=ctx, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    class PortableImageProcessor(ImageProcessor):
        """Upstream ImageProcessor, but the aligner runs on CUDA or CPU (MPS lacks svd and the
        fp16 warp is not worth it) and every detection's landmarks are kept for ref selection."""

        def __init__(self, resolution=512, device="cpu", mask_image=None):
            self.resolution = resolution
            self.resize = transforms.Resize((resolution, resolution),
                                            interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
            on_cuda = device.startswith("cuda")
            self.restorer = AlignRestore(resolution=resolution, device=device if on_cuda else "cpu",
                                         dtype=torch.float16 if on_cuda else torch.float32)
            self.mask_image = load_fixed_mask(resolution) if mask_image is None else mask_image
            self.face_detector = PortableFaceDetector(device=device)
            self.landmarks: list = []

        def affine_transform(self, image):
            bbox, lmk = self.face_detector(image)
            if bbox is None:
                raise RuntimeError(f"Face not detected in frame {len(self.landmarks)}")
            self.landmarks.append((bbox, lmk))
            return self.affine_transform_from(image, lmk)

        def affine_transform_from(self, image, landmark_2d_106):
            # upstream ImageProcessor.affine_transform after detection, unchanged
            import cv2

            pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
            pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)
            pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)
            landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])
            face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
            box = [0, 0, face.shape[1], face.shape[0]]
            face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
            face = rearrange(torch.from_numpy(face), "h w c -> c h w")
            return face, box, affine_matrix

    class PortableLipsyncPipeline(LipsyncPipeline):
        deterministic_vae = False

        # -- VAE encode, optionally deterministic (parity runs)
        def _encode(self, x, generator):
            dist = self.vae.encode(x).latent_dist
            z = dist.mode() if self.deterministic_vae else dist.sample(generator=generator)
            return (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        def prepare_mask_latents(self, mask, masked_image, height, width, dtype, device, generator, cfg):
            mask = torch.nn.functional.interpolate(mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
            masked_image_latents = self._encode(masked_image.to(device=device, dtype=dtype), generator).to(device=device, dtype=dtype)
            mask = rearrange(mask.to(device=device, dtype=dtype), "f c h w -> 1 c f h w")
            masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")
            if cfg:
                mask, masked_image_latents = torch.cat([mask] * 2), torch.cat([masked_image_latents] * 2)
            return mask, masked_image_latents

        def prepare_image_latents(self, images, device, dtype, generator, cfg):
            z = rearrange(self._encode(images.to(device=device, dtype=dtype), generator), "f c h w -> 1 c f h w")
            return torch.cat([z] * 2) if cfg else z

        def restore_video(self, faces, video_frames, boxes, affine_matrices):
            rdev = self.image_processor.restorer.device
            out = []
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = boxes[i]
                face = torchvision.transforms.functional.resize(
                    face.float(), size=(int(y2 - y1), int(x2 - x1)),
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=True).to(rdev)
                out.append(self.image_processor.restorer.restore_img(video_frames[i], face, affine_matrices[i]))
            return np.stack(out, axis=0)

        def align_all(self, frames):
            faces, boxes, mats, opn, usable = [], [], [], [], []
            for f in frames:
                face, box, m = self.image_processor.affine_transform(f)
                bbox, lmk = self.image_processor.landmarks[-1]
                h = max(bbox[3] - bbox[1], 1)
                opn.append(refmodes.mouth_openness(lmk.astype(float), h))
                eyes = (np.mean(lmk[[43, 48, 49, 51, 50]], 0) + np.mean(lmk[101:106], 0)) / 2
                eye_d = np.linalg.norm(np.mean(lmk[[43, 48, 49, 51, 50]], 0) - np.mean(lmk[101:106], 0)) + 1e-6
                nose = np.mean(lmk[[74, 77, 83, 86]], 0)
                usable.append(abs(nose[0] - eyes[0]) / eye_d < 0.25)  # roughly frontal
                faces.append(face); boxes.append(box); mats.append(m)
            return torch.stack(faces), boxes, mats, np.array(opn), np.array(usable)

        @torch.no_grad()
        def run(self, *, video_path, audio_path, work_dir, num_frames, resolution, steps, guidance, dtype,
                mask_image_path, ref, generator, dump=None, dump_chunks=1, progress=print):
            device = self._execution_device
            report = {"chunks": []}
            self.image_processor = PortableImageProcessor(resolution, device=str(device) if device.type == "cuda" else device.type,
                                                          mask_image=load_fixed_mask(resolution, mask_image_path))
            cfg = guidance > 1.0
            self.scheduler.set_timesteps(steps, device=device)
            timesteps = self.scheduler.timesteps
            extra = self.prepare_extra_step_kwargs(generator, 0.0)

            t0 = time.time()
            feat = self.audio_encoder.audio2feat(str(audio_path))
            chunks_a = self.audio_encoder.feature2chunks(feature_array=feat, fps=25)
            from latentsync.utils.util import read_video
            frames = read_video(str(video_path), change_fps=False, use_decord=False)
            if dump:
                dump.put("audio/whisper_feature", feat)
            report["t_audio_video_s"] = round(time.time() - t0, 2)

            t0 = time.time()
            n_src = len(frames)
            n = len(chunks_a)
            if n > n_src:
                progress(f"[warn] audio needs {n} frames, video has {n_src}: upstream ping-pong loops the video "
                         f"(back-and-forth motion). For an uncut take give a clip at least as long as the voice.")
                order = []
                k = 0
                while len(order) < n:
                    seq = list(range(n_src)) if k % 2 == 0 else list(range(n_src))[::-1]
                    order += seq
                    k += 1
                order = order[:n]
            else:
                order = list(range(n))
            faces_src, boxes_src, mats_src, opn_src, use_src = self.align_all(frames[:max(order) + 1])
            idx = torch.tensor(order)
            faces = faces_src[idx]
            boxes = [boxes_src[i] for i in order]
            mats = [mats_src[i] for i in order]
            video_frames = frames[order]
            ref_idx = refmodes.ref_indices(ref["mode"], len(faces_src), num_frames, idx=ref.get("idx"),
                                           shift=ref.get("shift"), openness=opn_src, usable=use_src)
            ref_idx = [ref_idx[i] for i in order]
            report["t_align_s"] = round(time.time() - t0, 2)
            report["ref"] = {**ref, "first_indices": ref_idx[:32], "unique": len(set(ref_idx))}
            report["source_mouth_openness"] = {"mean": float(opn_src.mean()), "min": float(opn_src.min()),
                                               "max": float(opn_src.max()), "frontal_share": float(use_src.mean())}
            if dump:
                dump.put("source/openness", opn_src)
                dump.put("source/ref_idx", np.array(ref_idx))

            all_latents = self.prepare_latents(n, self.vae.config.latent_channels, resolution, resolution, dtype, device, generator)
            if dump:
                dump.put("noise/all_latents", all_latents)

            out_faces = []
            t0 = time.time()
            n_inf = math.ceil(n / num_frames)
            for i in range(n_inf):
                s, e = i * num_frames, min((i + 1) * num_frames, n)
                d = dump if (dump and i < dump_chunks) else None
                a = torch.stack(chunks_a[s:e]).to(device, dtype=dtype)
                audio_embeds = torch.cat([torch.zeros_like(a), a]) if cfg else a
                tgt = faces[s:e]
                pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(tgt, affine_transform=False)
                if ref["mode"] == "same":
                    ref_pixel_values = pixel_values
                else:
                    ref_pixel_values, _, _ = self.image_processor.prepare_masks_and_masked_images(
                        faces_src[torch.tensor(ref_idx[s:e])], affine_transform=False)
                mask_lat, masked_lat = self.prepare_mask_latents(masks, masked_pixel_values, resolution, resolution,
                                                                 dtype, device, generator, cfg)
                ref_lat = self.prepare_image_latents(ref_pixel_values, device, dtype, generator, cfg)
                latents = all_latents[:, :, s:e]
                if d:
                    for k_, v_ in {"faces": tgt, "audio_chunks": a, "pixel_values": pixel_values,
                                   "masked_pixel_values": masked_pixel_values, "masks": masks,
                                   "ref_pixel_values": ref_pixel_values, "mask_latents": mask_lat,
                                   "masked_image_latents": masked_lat, "ref_latents": ref_lat,
                                   "latents_init": latents}.items():
                        d.put(f"chunk{i}/{k_}", v_)
                sens = []
                for j, t in enumerate(timesteps):
                    x = torch.cat([latents] * 2) if cfg else latents
                    x = self.scheduler.scale_model_input(x, t)
                    x = torch.cat([x, mask_lat, masked_lat, ref_lat], dim=1)
                    if d and j == 0:
                        hooks = d.hook_unet(self.unet, f"chunk{i}/step0")
                    eps = self.unet(x, t, encoder_hidden_states=audio_embeds).sample
                    if d and j == 0:
                        for h in hooks:
                            h.remove()
                        d.put(f"chunk{i}/step0/unet_input", x)
                        d.put(f"chunk{i}/step0/t", t)
                        d.put(f"chunk{i}/step0/audio_embeds", audio_embeds)
                        d.put(f"chunk{i}/step0/noise_pred", eps)
                    if cfg:
                        eu, ea = eps.chunk(2)
                        sens.append(float((ea - eu).float().norm() / (eu.float().norm() + 1e-8)))
                        eps = eu + guidance * (ea - eu)
                    latents = self.scheduler.step(eps, t, latents, **extra).prev_sample
                dec = self.decode_latents(latents)
                if d:
                    d.put(f"chunk{i}/latents_final", latents)
                    d.put(f"chunk{i}/decoded", dec)
                # paste the untouched surroundings of the TARGET frame back (upstream passes ref here)
                dec = self.paste_surrounding_pixels_back(dec, pixel_values, 1 - masks, device, dtype)
                out_faces.append(dec)
                c = {"chunk": i, "frames": [s, e],
                     "audio_sensitivity": round(sum(sens) / len(sens), 4) if sens else None}
                report["chunks"].append(c)
                progress(f"chunk {i + 1}/{n_inf} audio_sensitivity={c['audio_sensitivity']}  ({time.time() - t0:.0f}s)")
            report["t_denoise_s"] = round(time.time() - t0, 2)
            t0 = time.time()
            synced = self.restore_video(torch.cat(out_faces), video_frames, boxes, mats)
            report["t_restore_s"] = round(time.time() - t0, 2)
            sv = [c["audio_sensitivity"] for c in report["chunks"] if c["audio_sensitivity"] is not None]
            report["audio_sensitivity_mean"] = round(sum(sv) / len(sv), 4) if sv else None
            return synced, report

    return PortableLipsyncPipeline


# ----------------------------------------------------------------------------- tensor dumps
class Dumper:
    """Writes named tensors as float32 .npy (+ manifest) so another implementation can be compared."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = {}

    def put(self, name, value):
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            arr = value.detach().float().cpu().numpy()
        else:
            arr = np.asarray(value)
            if arr.dtype.kind == "f":
                arr = arr.astype("float32")
        p = self.root / (name + ".npy")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        self.manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    def hook_unet(self, unet, prefix):
        hooks = []

        def mk(name):
            def fn(_m, _i, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                if hasattr(o, "sample"):
                    o = o.sample
                self.put(f"{prefix}/unet/{name}", o)
            return fn

        for k, m in [("conv_in", unet.conv_in), ("mid_block", unet.mid_block), ("conv_out", unet.conv_out)] + \
                [(f"down_blocks.{i}", b) for i, b in enumerate(unet.down_blocks)] + \
                [(f"up_blocks.{i}", b) for i, b in enumerate(unet.up_blocks)]:
            hooks.append(m.register_forward_hook(mk(k)))
        return hooks

    def close(self, extra=None):
        (self.root / "manifest.json").write_text(json.dumps({"tensors": self.manifest, **(extra or {})}, indent=1))


# ----------------------------------------------------------------------------- build + run
def load_unet(config_model: dict, ckpt: Path):
    import torch
    from latentsync.models.unet import UNet3DConditionModel

    unet = UNet3DConditionModel.from_config(config_model)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    res = unet.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    audio_missing = [k for k in missing if any(h in k for h in AUDIO_KEY_HINTS)]
    rep = {"checkpoint": str(ckpt), "n_keys": len(sd), "missing": len(missing), "unexpected": len(unexpected),
           "missing_sample": missing[:12], "unexpected_sample": unexpected[:12], "audio_keys_missing": len(audio_missing)}
    if audio_missing:
        raise SystemExit(f"checkpoint {ckpt} lacks {len(audio_missing)} audio cross-attention weights "
                         f"(e.g. {audio_missing[:3]}); the model would ignore the voice. Wrong version/config?")
    return unet, rep


def run(args) -> dict:
    t_start = time.time()
    video = Path(args.video).expanduser().resolve()
    audio = Path(args.audio).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ls_dir = Path(args.latentsync_dir or config.latentsync_dir()).expanduser().resolve()
    cfg_rel, resolution, default_g = config.VERSIONS[args.version]
    ckpt = Path(args.ckpt).expanduser().resolve() if args.ckpt else config.checkpoint(args.version, ls_dir)
    whisper = config.whisper_checkpoint(args.version, ls_dir)
    for p, what in ((video, "video"), (audio, "audio"), (ckpt, f"LatentSync {args.version} UNet"), (whisper, "whisper tiny.pt")):
        if not p.exists():
            raise SystemExit(f"{what} not found: {p}")
    rev = import_upstream(ls_dir)

    import numpy as np
    import torch
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDIMScheduler
    from omegaconf import OmegaConf
    from latentsync.whisper.audio2feature import Audio2Feature

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    audio_device = args.audio_device if args.audio_device != "auto" else ("cuda" if device.startswith("cuda") else "cpu")
    conf = OmegaConf.load(cfg_rel)
    if args.resolution and args.resolution != resolution:
        print(f"[warn] {args.version} is trained at {resolution}px; ignoring --resolution {args.resolution}", file=sys.stderr)
    guidance = default_g if args.guidance is None else args.guidance
    seed = args.seed

    work = Path(tempfile.mkdtemp(prefix="ls_official_"))
    v25 = work / "video25.mp4"
    subprocess.run([config.ffmpeg(), "-loglevel", "error", "-y", "-nostdin", "-i", str(video), "-r", "25",
                    "-an", "-crf", "16", str(v25)], check=True)
    a16 = work / "audio16k.wav"
    subprocess.run([config.ffmpeg(), "-loglevel", "error", "-y", "-nostdin", "-i", str(audio), "-ac", "1",
                    "-ar", "16000", str(a16)], check=True)

    print(f"device={device} dtype={str(dtype).split('.')[-1]} whisper_on={audio_device} version={args.version} "
          f"res={resolution} steps={args.steps} guidance={guidance} ref={args.ref_mode} seed={seed} upstream={rev[:10]}")
    set_seed(seed)
    scheduler = DDIMScheduler.from_pretrained("configs")
    audio_encoder = Audio2Feature(model_path=str(whisper), device=audio_device, num_frames=conf.data.num_frames,
                                  audio_feat_length=conf.data.audio_feat_length)
    vae_src = os.environ.get("ULTRON_LATENTSYNC_VAE", "stabilityai/sd-vae-ft-mse")
    vae = AutoencoderKL.from_pretrained(vae_src, torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, key_report = load_unet(OmegaConf.to_container(conf.model), ckpt)
    unet = unet.to(dtype=dtype)
    Pipe = make_classes()
    pipe = Pipe(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to(device)

    dump = Dumper(Path(args.dump)) if args.dump else None
    deepcache = args.deepcache and not args.parity
    if args.parity:
        pipe.deterministic_vae = True
    if deepcache:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    gen = torch.Generator(device="cpu").manual_seed(seed) if (args.parity or device == "mps") else None

    ref = {"mode": args.ref_mode, "idx": args.ref_idx, "shift": args.ref_shift}
    frames, rep = pipe.run(video_path=v25, audio_path=a16, work_dir=work, num_frames=conf.data.num_frames,
                           resolution=resolution, steps=args.steps, guidance=guidance, dtype=dtype,
                           mask_image_path=conf.data.mask_image_path, ref=ref, generator=gen, dump=dump,
                           dump_chunks=args.dump_chunks)
    from latentsync.utils.util import write_video
    silent = work / "video.mp4"
    write_video(str(silent), frames, fps=25)
    dur = len(frames) / 25.0
    subprocess.run([config.ffmpeg(), "-loglevel", "error", "-y", "-nostdin", "-i", str(silent), "-i", str(audio),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-t", f"{dur:.3f}", str(out)], check=True)
    rep.update({
        "out": str(out), "video": str(video), "audio": str(audio), "backend": "official", "upstream_commit": rev,
        "version": args.version, "resolution": resolution, "steps": args.steps, "guidance": guidance, "seed": seed,
        "device": device, "dtype": str(dtype).split(".")[-1], "deepcache": deepcache, "parity": bool(args.parity),
        "weights": key_report, "frames": int(len(frames)), "seconds_total": round(time.time() - t_start, 1),
    })
    if dump:
        dump.close({"report": rep})
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("out", "audio_sensitivity_mean", "seconds_total")}))
    return rep


def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="1.6", choices=sorted(config.VERSIONS))
    ap.add_argument("--guidance", type=float, default=None, help="CFG scale (upstream 1.5; long takes often want 2-3)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--resolution", type=int, default=None, help="ignored; fixed by --version")
    ap.add_argument("--ref-mode", default=os.environ.get("LS_REF_MODE", "same"), choices=refmodes.MODES)
    ap.add_argument("--ref-idx", default=os.environ.get("LS_REF_IDX"), help="frame for --ref-mode fixed, or 'auto'")
    ap.add_argument("--ref-shift", type=int, default=int(os.environ["LS_REF_SHIFT"]) if os.environ.get("LS_REF_SHIFT") else None)
    ap.add_argument("--device", default="auto", help="auto|cuda|cuda:N|mps|cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"])
    ap.add_argument("--audio-device", default="auto", help="whisper device (Mac default cpu)")
    ap.add_argument("--deepcache", action=argparse.BooleanOptionalAction, default=True,
                    help="upstream inference.sh enables it; off for parity runs")
    ap.add_argument("--latentsync-dir", default=None)
    ap.add_argument("--ckpt", default=None, help="override UNet checkpoint path")
    ap.add_argument("--dump", default=None, help="write intermediate tensors here (for parity.py)")
    ap.add_argument("--dump-chunks", type=int, default=1)
    ap.add_argument("--parity", action="store_true", help="deterministic run for layer-by-layer comparison")
    ap.add_argument("--report", default=None, help="write report json here (default: <out>.report.json)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    args = ap.parse_args(argv)
    # progress lines go to a log file when run.py/clone.py call us; line-buffer so the log shows them live
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    if args.ref_idx not in (None, "auto"):
        args.ref_idx = int(args.ref_idx)
    if args.report is None:
        args.report = str(Path(args.out).expanduser().resolve()) + ".report.json"
    run(args)


if __name__ == "__main__":
    main()
