# Architecture

```
            stills (Qwen-Image / Qwen-Image-Edit)             music (ACE-Step)
                     |                                              |
   storyboard -> keyframe per shot -> LTX-2.5 / H3 image-to-video -> take review (contact sheets)
                                                                      |
   voice (Fish / Chatterbox) -> native speech or voice conversion -> QA gates -> edit (ffmpeg / HyperFrames) -> loudnorm -> phone copy
```

* **Studio server** (`src/studio/server.py`): an MCP server exposing `generate_image`, `edit_image`, `generate_video`, `generate_music`, `generate_speech`, `transcribe`, `lipsync`, `make_reel`, `render_hyperframes`, `mux_audio`, `free_memory`, `job_status`... Heavy jobs run one at a time in a queue. It talks to **mlx-serve** on 127.0.0.1:11234 for the ddalcu/* models and to the local-avatar venv for voice, ASR and lip-sync.
* **LTX runner**: [xocialize/ltx-2-mlx](https://github.com/xocialize/ltx-2-mlx) (commit pinned in the catalog). Weights are assembled into a symlink "pack" folder by `ultron.py` (int8 transformer from `mlx-community/ltx-2.5-mlx-ditq8` plus everything else from `mlx-community/ltx-2.5-mlx-q8`; do not download the 100 GB base repo).
* **Tools** (`src/tools`, rendered to `runtime/tools`):
  | tool | job |
  |---|---|
  | `heavy_guard.sh` | refuses to start a heavy job if another runs, studio models are loaded, or memory > 55 GB |
  | `ramlog.sh` | samples memory to a CSV so peaks are measured, not guessed |
  | `clip_qa.py` | gate for talking clips: words by character overlap (Latin and Devanagari), face match, calm head, props, audio jumps |
  | `face_sim.py`, `facelock.py` | ArcFace identity score of a clip against a reference |
  | `sync_metric.py` | lip-sync metric (advisory for non-English) |
  | `voiceconvert.py`, `voicefit2.py`, `tighten.py`, `trim_pauses.py` | voice conversion, phrase-level fit with cuts only in quiet parts, pause removal without audio breaks |
  | `asr_words.py` | Whisper word timestamps |
  | `finish.sh` | render + loudnorm + phone copy |
* **Pipelines** (`src/pipelines`): runnable examples. `build_priest.py` and `build_gojo2.py` cut a fight to a track's strongest percussive onsets; `shots.py` / `lab.py` show the action-prompt format; `make_assets5.py` converts native LTX speech to a cloned voice.
* **Skills** (`skills/`): the lessons, written for Claude Code. They are plain Markdown and worth reading even without Claude.
