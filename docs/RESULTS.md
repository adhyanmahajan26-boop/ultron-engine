# Measured results (our logs, one machine)

| thing | number |
|---|---|
| LTX compute per second of footage | about 30 s (H3 about 95 s) |
| LTX 4 s clip | about 110 s |
| H3 4 s clip | median 20 min over 64 logged jobs, about 7 min on a clean run |
| LatentSync | about 14 min per 10 s of talking |
| Hindi talking film, 76 s | 3 h 28 min with this engine vs about 10 h 40 min for the previous version of the same film (from the build logs) |
| 48 s anime fight (30 LTX takes + keyframes + edit) | 1 h 40 min; redoing the 6 action shots on H3 took 80 min more |
| 44 s live-action trailer (26 LTX takes, prompt test, edit) | 1 h 38 min |
| Face match (ArcFace cosine, one subject) | H3 0.72-0.87, LTX 0.68-0.79, still plus LatentSync 0.89-0.94 |

Findings: LTX is the best default (fast, speaks natively, lips match); H3 was clearer for anime action choreography (cause and effect between characters) at about 6x the time; hybrid is best. Single-subject, non-chaotic shots work; mobs, crowds and fast multi-character choreography still morph.
