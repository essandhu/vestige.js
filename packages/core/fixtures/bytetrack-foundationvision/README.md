# `bytetrack-foundationvision/`

Cross-implementation oracle for `ByteTracker`, generated against
[FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) —
the official implementation from the ECCV 2022 paper (Zhang et al.,
arXiv:2110.06864).

Third and last of the sister fixtures, after `ocsort-noahcao/` (PR #14) and
`sort-abewley/` (PR #15). Consumed by
`tests/validation/bytetrack-foundationvision-fixture.test.ts`.

| | |
|---|---|
| Reference | `FoundationVision/ByteTrack` |
| Pinned commit | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` |
| Reference file | `yolox/tracker/byte_tracker.py` (+ `matching.py`, `kalman_filter.py`, `basetrack.py`) |
| Config | `track_thresh=0.5`, `track_buffer=30`, `match_thresh=0.8`, `mot20=False`, `frame_rate=30` |
| Sequence | 90 frames, 8 tracks, 80×80 boxes |

## Setup

Heavier than the other two sister fixtures: `byte_tracker.py` lives inside the
YOLOX framework rather than being a standalone file like `sort.py`.

```powershell
git clone https://github.com/FoundationVision/ByteTrack.git ~/repos/ByteTrack
git -C ~/repos/ByteTrack checkout d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
cd packages/core/fixtures
pip install -r requirements.txt   # adds lapx + cython_bbox for this fixture
py bytetrack-foundationvision/gen.py
```

`gen.py` reads `BYTETRACK_PATH` (default `~/repos/ByteTrack`). It does not need
YOLOX installed — see the import-rigging comment block in the script for how it
loads only the four files in `yolox/tracker/` while sidestepping numpy 2's
removal of `np.float` and YOLOX's eager `torch.distributed` import.

Only the *generation* step needs any of this. `pnpm test` reads the committed
`data.json`; no Python required.

## Sequence design

Eight tracks, each isolated in its own lane, velocities ≤ 2 px/frame so no two
tracks ever overlap. Each targets a specific lifecycle edge:

| Track | Behavior | Edge under test |
|---|---|---|
| A | always visible, score 0.9 | stage-1 control |
| B | score oscillates 0.9 / 0.4 | stage-2 recovery of a **confirmed** track (stays output every frame) |
| C | 10-frame occlusion | confirmed → lost → stage-1 re-activation, **same id** |
| D | **30**-frame occlusion | `max_time_lost` boundary: tsu reaches 30, not `> 30` → **survives**, same id |
| E | **32**-frame occlusion | the other side: tsu passes 31 → **removed** → reappearance spawns a **new id** |
| F | spawns on frame 5 | tentative → **stage-3** promotion; *not* output on its spawn frame |
| G | frame 12, gone, **returns frame 40** | the one-chance rule for tentatives (see below) |
| H | score 0.05 on f31; **exactly 0.5** on f51 | the two score-band boundaries |

Two details are load-bearing and worth not "simplifying" later:

**G's reappearance is not decoration.** A tentative track is never exported, so a
tracker that simply *failed* to remove a stale tentative would look identical on
every frame it was absent — the bug would be invisible. Bringing the detection
back at frame 40 makes it observable: correctly, G is long gone and a **new id**
spawns (confirmed frame 42); with a lingering tentative, the stale G wins the
stage-3 match and reappears under its **old id**, a frame early. Verified by
mutation — deleting the `state = 'removed'` line in `bytetrack.ts` step 6 fails
this fixture only *because* G comes back.

**H's frame-51 detection scores exactly `track_thresh`.** The reference's bands
are `remain = scores > track_thresh` and `second = (scores > 0.1) & (scores <
track_thresh)`, so a detection scoring exactly `track_thresh` falls into neither
and is silently discarded. That quirk is documented in `ByteTrackerOptions`;
this frame is what proves vestige actually reproduces it.

## Known divergences the sequence deliberately avoids

The disjoint-lane design is not just tidiness — it keeps the sequence inside the
envelope where vestige and the reference provably agree. Two known divergences
sit outside it, and **this fixture does not cover them**:

1. **IoU pixel convention.** `matching.ious()` goes through `cython_bbox`, which
   uses the Faster R-CNN `+1` convention (`(x2-x1+1)*(y2-y1+1)`); vestige's
   `geometry/iou.ts` uses the standard one. Two half-overlapping 10×10 boxes score
   0.375 there vs 0.3333 here. With disjoint lanes every cost is either ≈0 (a track
   and its own detection) or 1.0 (everything else), so the discrepancy cannot flip
   an assignment.

2. **The reference's one-frame track-resurrection bug.** In `byte_tracker.py:283-284`
   the bookkeeping runs `self.lost_stracks = sub_stracks(self.lost_stracks,
   self.removed_stracks)` *before* `self.removed_stracks.extend(removed_stracks)`,
   so a track marked removed on frame N is still in `lost_stracks` — and therefore
   still in `strack_pool` — on frame N+1, where a reappearing detection can
   `re_activate()` it under its original id. (The object ends up in
   `tracked_stracks` and `removed_stracks` simultaneously.) The reference's
   re-association window is thus `L+1 .. L+32`; vestige's is `L+1 .. L+31`.
   Track E steps one frame clear of that window so that D/E test the
   `max_time_lost` boundary itself rather than the bug.

Both are recorded, with the other confirmed divergences, in
`docs/decisions/0003-tracker-lifecycle-bookkeeping.md` §6. Do not "fix" this
fixture by widening `BBOX_TOLERANCE` or by moving E back one frame — if the
validation test starts failing, the divergence is real.

## Regenerating

`gen.py` and `data.json` are committed together, never one without the other
(ADR-0002 §4). Regeneration is only needed if the sequence changes or the
reference commit is re-pinned.
