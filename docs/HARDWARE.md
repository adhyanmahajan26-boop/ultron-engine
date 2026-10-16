# Hardware

Measured on an Apple M3 Ultra with 96 GB unified memory (macOS). We have not measured other machines; the rest is arithmetic, labelled as such.

| job | peak memory we measured |
|---|---|
| LTX 4 s clip | about 52 GB system memory |
| LTX 10 s clip | 85 GB |
| LTX keyframe shot | about 51 GB |
| Qwen-Image-Edit 2511 | about 39 GB |
| Qwen-Image 2.1 resident | about 22 GB |
| H3 8-bit resident | about 70 GB |

Implications (not measured): with 32-64 GB, expect LTX 3-4 s clips to be tight or to swap, never run two heavy jobs together, and prefer `mixed` (video in the cloud, the rest local). On non-Apple hardware the local models do not run at all (MLX): use `cloud`.
Check memory with `top -l 1 -o mem` (includes compressed memory) and sample with `runtime/tools/ramlog.sh`.
