# 0007 — OC-SORT exports observations, and clean fixtures cannot see it

- **Date:** 2026-07-13
- **Status:** Accepted
- **Scope:** `OcSortTracker.exportConfirmed`'s output bbox, the OCR stage's never-observed-track gate, and the `asoFunc: 'giou'` / `useByte: true` option paths. Introduces `fixtures/detection-noise/` and `fixtures/ocsort-giou-byte/`.

## Context

`noahcao/OC_SORT` does not export the box its Kalman filter believes in. It exports the
last box the **detector** gave it (`ocsort.py:313-320`):

```python
if trk.last_observation.sum() < 0:
    d = trk.get_state()[0]        # never observed -> Kalman state
else:
    d = trk.last_observation[:4]  # the RAW DETECTION
```

This is the one place OC-SORT parts company with SORT and ByteTrack, which both export
their filter state (`sort.py` via `get_state()`, ByteTrack via `STrack.tlbr`). It is not
an implementation accident. OC-SORT is *observation-centric* — the whole paper is about
Kalman drift during occlusion, and ORU and OCM exist to correct for it. Having spent that
effort, it declines to hand you the drifted box.

vestige exported the Kalman posterior. The divergence was not even unknown: the JSDoc on
`exportConfirmed` quoted the reference's rule and then dismissed it —

> *"matched here by exporting `track.bbox` — equal-by-design to the last observation within Kalman-filter noise."*

## Why every fixture agreed with a false claim

That sentence is true in exactly one regime, and it is the regime every fixture in this
repo used: **noise-free detections on perfectly linear trajectories.** Feed a Kalman
filter clean measurements of constant-velocity motion and its posterior converges onto
the measurement — not approximately, *numerically*. Measured on the existing
`ocsort-noahcao` sequence, the delta between the two conventions is **0.000000 px**.

So a tracker exporting the Kalman state and one exporting the raw detection emitted
byte-identical output, the cross-implementation fixture passed, and the claim in the
JSDoc read as verified. A Kalman posterior differing from its measurement is the entire
*point* of running one; our fixtures had quietly engineered that difference away.

Add σ = 4 px of Gaussian localization noise — a modest, ordinary detector error — to the
*same sequence*, and the two conventions diverge immediately: **0.8 px on frame 2, up to
~9 px after an occlusion.** Every exported box was wrong on real data. It would have
landed as a MOTP and HOTA-LocA deficit against the paper, and it would have looked like a
Kalman bug.

This is ADR-0005's lesson recurring in a new place, and it is worth stating in its
general form:

> **A fixture tests the regime its inputs put the code in. Noise-free synthetic data is
> not a neutral simplification — it is a regime, and it is one in which several distinct
> implementations become observationally identical.**

## Decision

### 1. Export the last observation (fixed)

`exportConfirmed` now emits `lastObservation` when the track has one, falling back to the
Kalman-derived `track.bbox` otherwise. The fallback is reachable, not defensive padding:
`lastObservation` stays `null` until a track's first *match* (spawning does not set it,
mirroring `KalmanBoxTracker.__init__` leaving the `[-1,-1,-1,-1,-1]` sentinel in place),
and during the `frameIndex <= minHits` warmup such a track is exported anyway.

`fixtures/detection-noise/` drives **all three** references through one noisy sequence.
That breadth is the point: SortTracker and ByteTracker **pass it unchanged**, because
their references really do export filter state. Only OC-SORT moved. A fixture that drove
only the tracker already under suspicion would not have been evidence of anything.

### 2. Do NOT reproduce the OCR placeholder match (deliberate divergence)

The reference's OCR stage builds its track side from `last_boxes[unmatched_trks]`
(`ocsort.py:281`) — which **includes never-observed tracks, at their `[-1,-1,-1,-1]`
placeholder box** — and scores them with `asso_func`.

Under `iou` that placeholder has zero area and scores 0.0000 against everything, so it can
never match and the question never arises. That is why the default config never exposed
it. Under `giou` the score is driven by the *enclosing* box, and a placeholder sitting at
(-1, -1) is near the image origin:

| detection | `giou` (normalized) |
|---|---|
| `[2, 2, 82, 82]` | **0.4645** → matches (> 0.3) |
| `[300, 300, 380, 380]` | 0.0220 |
| `[600, 600, 680, 680]` | 0.0069 |

Executed against the reference: a track spawned at (600,600), then a lone detection at
(2,2) on the next frame —

```
frame 1: det [600,600,680,680] -> id 1 @ [600,600,680,680]
frame 2: det [2,2,82,82]       -> id 1 @ [2,2,82,82]        <- teleports 840 px
```

The track jumps the width of the frame and keeps its id. **We do not reproduce this.** OCR
exists to recover a track from its last real observation; a track with no observation has
nothing to recover *from*, and matching against the placeholder is matching against
garbage that happens to sit near the origin. vestige excludes never-observed tracks from
the OCR pool.

Same posture as ADR-0003 §7's ByteTrack resurrection: keep vestige correct, document the
divergence, and pin it (`tests/unit/ocsort-ocr-sentinel.test.ts`) so a future "fix" toward
the reference fails loudly. The pinning test also asserts that a genuinely *observed*
track is still recoverable by OCR — otherwise the divergence could be "resolved" by
gutting the stage.

### 3. Cover `giou` and `useByte` (new fixture)

Both are documented, exported options that had **zero** reference-backed coverage: every
other fixture ran `asso_func='iou'`, `use_byte=False`. `fixtures/ocsort-giou-byte/` closes
that.

Designing it discriminating took a second attempt, which is the instructive part. The
first sequence passed on the first run *and survived every mutation* — breaking the giou
normalization, forcing the BYTE/OCR stages back to plain IoU, and disabling OCR entirely
all left it green. The arithmetic explains why. For two `BOX`-sized squares offset by `d`
with no overlap:

```
iou             = 0                  (flat, for every d >= BOX)
giou_normalized = BOX / (BOX + d)    (smooth in the separation)
```

**`giou` differs from `iou` exactly where the boxes do not overlap.** Every recovery in
the first sequence overlapped its target — the one regime where the two agree. And OCR
never fired at all, because under linear motion the Kalman prediction stays accurate
through an occlusion, so the primary stage always matched first and OCR was never reached.

The fix was two probes built on **stationary** tracks (so the Kalman prediction and the
last observation coincide) with detections displaced 120 px — invisible to the primary
stage, which hard-codes `iou_batch` (0.000), but reachable by whichever stage uses
`asso_func` (0.400). One probes the BYTE stage via a low score, one probes OCR via an
occlusion. Both now have a *different outcome* under `giou` than under `iou`, and all four
mutations are caught.

Also confirmed rather than assumed: the reference **normalizes** `giou` to `[0,1]`
(`association.py:54`), so `iou_threshold = 0.3` keeps meaning "30% score". Had it not, the
threshold would have meant something entirely different and every giou association would
have been wrong. The port already did this correctly.

## Consequences

- **Every OC-SORT box changes on real data.** MOTP and HOTA's LocA move; any OC-SORT
  number produced before this is void. Landing it before the first MOT17 run is deliberate.
- The clean-regime fixtures (`ocsort-noahcao/`) still pass **unchanged**, which is exactly
  the tell: they never distinguished the two conventions.
- `giou` and `useByte` are now reference-backed for the first time.

## Things deliberately deferred

| Item | When to revisit |
|---|---|
| `getActiveTracks()` / `getLostTracks()` still materialize the Kalman-derived `bbox`. They are vestige-specific inspection APIs with no reference, and for a *lost* track the Kalman prediction (where we think it is *now*) is more useful than a stale observation. Only `update()`'s return follows the reference's rule. | If a user reports the inconsistency as surprising, or if a plugin needs one convention throughout. |
| Noise fixtures for SORT and ByteTrack beyond the parity check in `detection-noise/`. They pass today, but only their *export* path was under test there; noise could still interact with, say, SORT's scale-velocity safeguard. | With PR-D (degenerate boxes), which is where that safeguard is already under review. |
