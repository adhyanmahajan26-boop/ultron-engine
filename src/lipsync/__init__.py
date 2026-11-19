"""Lip-sync backends for Ultron Engine.

One interface, three ways to run LatentSync:
  * ``official`` - ByteDance's PyTorch code (pinned commit), in-process, CUDA / Apple MPS / CPU.
  * ``mlx``      - the unofficial Apple-MLX port in $ULTRON_AVATAR (scripts/lipsync.py).
  * ``cloud``    - a rented endpoint speaking the protocol in docs/CLOUD.md.

``backends.py`` and ``config.py`` are standard-library only so the studio server can import them
without torch. Everything that needs torch lives in ``official_runner.py`` and is imported lazily.
See docs/LIPSYNC.md.
"""

UPSTREAM_REPO = "https://github.com/bytedance/LatentSync"
UPSTREAM_COMMIT = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
