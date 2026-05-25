# ocsort-noahcao — cross-implementation fixture for OcSortTracker

Per-frame trace of `noahcao/OC_SORT` on a synthetic sequence, used by
`packages/core/tests/validation/ocsort-noahcao-fixture.test.ts` as the
faithfulness oracle for vestige.js's `OcSortTracker`.

## What this fixture asserts

> Run the same sequence through both implementations; outputs match per-frame
> within `1e-3` bbox tolerance, with a consistent track-id correspondence
> across all 60 frames.

Without this fixture, the scenario tests in `tests/validation/ocsort-synthetic.test.ts`
check what OC-SORT is *supposed* to do (hand-traced expected behavior).
This fixture checks what the canonical reference *actually* does, which is
the only test layer that catches a faithfulness regression where vestige
drifts from noahcao but still passes the hand-traced assertions.

## The sequence

60 frames, 6 tracks at well-separated start positions:

| Name | Start | Velocity | Occlusion frames (0-indexed) | Exercises |
|---|---|---|---|---|
| A | (50, 50) | (+5, 0) | 15–21 | ORU after 7-frame occlusion + smooth reappearance |
| B | (350, 50) | (0, +5) | — | Always-visible control |
| C | (50, 350) | (−3, +3) | 45–54 | 10-frame mid-motion occlusion |
| D | (350, 350) | (0, 0) | 30–41 | Stationary 12-frame occlusion (OCR territory: scale drift) |
| E | (600, 100) | (+2, +4) | — | Always-visible control |
| F | (100, 550) | (+8, −2) | 20–27 | ORU parallel to A's occlusion window |

All boxes are 80×80, no two ever IoU-overlap → no Hungarian tie-breaks → both
implementations spawn IDs in identical detection-index order, so the
`(vestige-id ↔ noahcao-id)` mapping is `1:1` and stable across the run.

The OC-SORT config is paper-default: `det_thresh=0.6, max_age=30, min_hits=3,
iou_threshold=0.3, delta_t=3, asso_func='iou', inertia=0.2, use_byte=False`.
Committed in the fixture envelope's `ocsort_config` block so the test reads
exactly what `gen.py` wrote.

## Regenerating data.json

One-time setup:

```powershell
# Clone the reference repo at the pinned commit (the same commit is recorded
# in the fixture envelope's `generator.noahcao_sha` field — keep them in sync).
git clone https://github.com/noahcao/OC_SORT.git ~/repos/OC_SORT
git -C ~/repos/OC_SORT checkout 8462e7e729a93ccd3bd995c0a79a890336cb3a0b

# In a venv, from packages/core/fixtures/:
pip install -r requirements.txt
```

To regenerate:

```powershell
python packages/core/fixtures/ocsort-noahcao/gen.py
```

The script reads `OC_SORT_PATH` (default `~/repos/OC_SORT`) and writes
`data.json` next to itself. Both `gen.py` and `data.json` are committed
together per ADR-0002 §4 — never one without the other.

## When to re-baseline

- **A noahcao update worth tracking.** Re-checkout the new commit, regenerate,
  inspect the JSON diff. Any per-frame output change is a deliberate
  re-baseline; record what changed in the commit message.
- **A vestige `OcSortTracker` change that affects observable output.** The
  test fails first; if the change is intentional, the JSON does not change
  (it's the reference), but the test message tells you which frame diverged
  and you investigate from there.

The 60-frame sequence is short enough (~72 KB JSON) that line-diffs in code
review are tractable. Larger sequences (MOT17-style) belong in
`packages/eval/` per ARCHITECTURE.md §10.3, not here.
