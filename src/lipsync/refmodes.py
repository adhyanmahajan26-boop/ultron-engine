"""Which source frame LatentSync gets as its unmasked *reference* image (numpy only, unit-tested).

Why this matters (see docs/LIPSYNC.md): upstream *training* takes the reference window from a
different, non-overlapping 16-frame stretch of the same video (latentsync/data/unet_dataset.py),
but upstream *inference* passes the very frame being re-drawn. The reference then already shows
the old mouth and the model can copy it instead of following the voice. Modes:

  same    upstream inference behaviour: ref[t] = t
  fixed   one frame for every t (idx, or "auto" = most closed mouth among usable frames)
  shift   ref[t] = (t + shift) mod n  (shift defaults to one chunk, 16 frames)
  window  training-like: for each 16-frame chunk use a whole other, non-overlapping 16-frame
          window, the one with the most closed mouth; falls back to fixed/auto for clips too
          short to have two windows
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MODES = ("same", "fixed", "shift", "window")


def mouth_openness(landmarks_106: np.ndarray, face_height: float) -> float:
    """Inner-lip opening normalised by face height (insightface 2d106: 64..71 = inner lips).
    Same definition as src/tools/sync_metric.py so numbers are comparable."""
    inner = landmarks_106[64:72]
    return float((inner[:, 1].max() - inner[:, 1].min()) / max(face_height, 1e-6))


def _usable(n: int, usable: Optional[Sequence[bool]]) -> np.ndarray:
    if usable is None:
        return np.ones(n, dtype=bool)
    u = np.asarray(usable, dtype=bool)
    return u if u.any() else np.ones(n, dtype=bool)


def auto_fixed_index(openness: Optional[Sequence[float]], usable: Optional[Sequence[bool]] = None) -> int:
    if openness is None or len(openness) == 0:
        return 0
    o = np.asarray(openness, dtype=float)
    u = _usable(len(o), usable)
    o = np.where(u, o, np.inf)
    return int(np.argmin(o))


def ref_indices(
    mode: str,
    n: int,
    chunk: int = 16,
    idx: Optional[int | str] = None,
    shift: Optional[int] = None,
    openness: Optional[Sequence[float]] = None,
    usable: Optional[Sequence[bool]] = None,
) -> list[int]:
    """Reference source-frame index for every target frame 0..n-1."""
    if n <= 0:
        return []
    if mode not in MODES:
        raise ValueError(f"ref mode must be one of {MODES}, got {mode!r}")
    if mode == "same":
        return list(range(n))
    if mode == "fixed":
        if idx is None or idx == "auto":
            k = auto_fixed_index(openness, usable)
        else:
            k = int(idx)
            if not 0 <= k < n:
                raise ValueError(f"LS_REF_IDX {k} outside 0..{n - 1}")
        return [k] * n
    if mode == "shift":
        s = chunk if shift is None else int(shift)
        return [(t + s) % n for t in range(n)]
    # window
    if n < 2 * chunk:
        return ref_indices("fixed", n, chunk, idx="auto", openness=openness, usable=usable)
    o = np.zeros(n) if openness is None else np.asarray(openness, dtype=float)
    u = _usable(n, usable)
    starts = np.arange(0, n - chunk + 1)
    # mean openness per candidate window, unusable frames make a window expensive
    cost = np.array([o[r:r + chunk].mean() + (0 if u[r:r + chunk].all() else 1e3) for r in starts])
    out: list[int] = []
    for s in range(0, n, chunk):
        span = min(chunk, n - s)
        ok = (starts + chunk <= s) | (starts >= s + chunk)  # non-overlapping, as in training
        if not ok.any():
            ok = starts != s
        r = int(starts[ok][np.argmin(cost[ok])])
        out.extend(r + k for k in range(span))
    return out


def from_env(env: dict) -> dict:
    """Read the MLX-port compatible env knobs (LS_REF_MODE, LS_REF_IDX, LS_REF_SHIFT)."""
    mode = env.get("LS_REF_MODE", "same") or "same"
    idx = env.get("LS_REF_IDX")
    sh = env.get("LS_REF_SHIFT")
    return {
        "mode": mode,
        "idx": (idx if idx in (None, "auto") else int(idx)),
        "shift": (int(sh) if sh not in (None, "") else None),
    }
