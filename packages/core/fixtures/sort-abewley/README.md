# sort-abewley — cross-implementation fixture for SortTracker

Per-frame trace of `abewley/sort` on a synthetic sequence, used by
`packages/core/tests/validation/sort-abewley-fixture.test.ts` as the
faithfulness oracle for vestige.js's `SortTracker`. Sister fixture to
`ocsort-noahcao/`: same harness, same envelope shape, different reference.

## What this fixture asserts

> Run the same sequence through both implementations; outputs match per-frame
> within `1e-3` bbox tolerance, with a consistent track-id correspondence
> across all 60 frames.

The scenario tests in `tests/unit/sort-tracker.test.ts` cover what SORT is
*supposed* to do in isolation; this fixture checks what `sort.py` *actually*
does on a multi-frame sequence — the only test layer that catches a
faithfulness regression where vestige drifts from the reference but still
passes the unit assertions.

## The sequence

60 frames, 6 tracks at well-separated start positions, default config
(`max_age=1, min_hits=3, iou_threshold=0.3`):

| Name | Start | Velocity | Occlusion frames (0-indexed) | Exercises |
|---|---|---|---|---|
| A | (50, 50) | (+5, 0) | — | Always-visible control |
| B | (350, 50) | (0, +5) | 11 | Single-frame miss; bug window at frames 13–14 |
| C | (50, 350) | (−3, +3) | 20, 35 | Two single-frame misses; two bug windows |
| D | (350, 350) | (0, 0) | — | Stationary control |
| E | (600, 100) | (+2, +4) | 25, 26 | Two consecutive misses → track removed → respawns with new id at frame 28 |
| F | (100, 550) | (+8, −2) | 45 | Late single-frame miss; bug window at frames 47–48 |

All boxes are 80×80, no two ever IoU-overlap → no Hungarian tie-breaks → both
implementations spawn IDs in identical detection-index order, so the
`(vestige-id ↔ abewley-id)` mapping is `1:1` and stable across the run
(except for E, whose re-spawn deliberately mints a new id on both sides).

## The bug window this fixture targets

With `max_age=1, min_hits=3`, `sort.py`'s output rule
(`sort.py:Sort.update`, line 245) is:

```python
(trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
```

After a single-frame miss on a previously-confirmed track:

```
Frame N:    matched, hit_streak=k (large), tsu=0
Frame N+1:  miss → tsu=1 → output gate `tsu<1` False → NOT output, kept
Frame N+2:  match → tsu=0, hit_streak=1
            → sort.py: 1 < min_hits(3) and frame_count > 3 → NOT output
            → buggy vestige (`state === 'confirmed'`)   → output (BUG)
Frame N+3:  hit_streak=2 → still bug
Frame N+4:  hit_streak=3 → both output, bug window closes
```

So each single-frame occlusion buys 2 frames of bug-exposing output
divergence. Tracks B (frames 13–14), C (frames 22–23 and 37–38), and F
(frames 47–48) each cover one such window; track E covers the orthogonal
"two-miss removal + respawn-with-new-id" path.

## Regenerating data.json

One-time setup:

```powershell
# Clone the reference repo at the pinned commit (the same commit is recorded
# in the fixture envelope's `generator.abewley_sha` field — keep them in sync).
git clone https://github.com/abewley/sort.git ~/repos/sort
git -C ~/repos/sort checkout 2236dff5019565958b84df7d871d41cc1db58ac7

# In a venv, from packages/core/fixtures/:
pip install -r requirements.txt
```

`gen.py` stubs out `matplotlib` and `skimage` (which `sort.py` imports only
for its CLI demo block) via `sys.modules`, so neither needs to be installed.

To regenerate:

```powershell
python packages/core/fixtures/sort-abewley/gen.py
```

The script reads `SORT_PATH` (default `~/repos/sort`) and writes `data.json`
next to itself. Both `gen.py` and `data.json` are committed together per
ADR-0002 §4 — never one without the other.

## When to re-baseline

- **An abewley update worth tracking.** Re-checkout the new commit, regenerate,
  inspect the JSON diff. Any per-frame output change is a deliberate
  re-baseline; record what changed in the commit message.
- **A vestige `SortTracker` change that affects observable output.** The
  test fails first; if the change is intentional, the JSON does not change
  (it's the reference), but the test message tells you which frame diverged
  and you investigate from there.
