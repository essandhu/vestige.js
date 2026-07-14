"""Drive noahcao/OC_SORT with asso_func='giou' AND use_byte=True.

Cross-implementation oracle for
`packages/core/tests/validation/ocsort-giou-byte-fixture.test.ts`.

Why this fixture exists
-----------------------
`asoFunc: 'giou'` and `useByte: true` are documented, exported options on
`OcSortTrackerOptions`, and until this landed NEITHER had a single line of
reference-backed coverage: every cross-implementation fixture in the repo ran
`asso_func='iou'`, `use_byte=False`. Two whole branches of shipped code, exported to
users, compared against nothing.

`giou` is not a cosmetic swap for `iou`. It changes:
  * the BYTE stage's cost matrix (`ocsort.py:262`, `self.asso_func(dets_second, u_trks)`),
  * the OCR stage's cost matrix (`ocsort.py:283`, `self.asso_func(left_dets, left_trks)`),
  * and therefore which tracks are recovered at all.
It does NOT change the primary stage: `association.py:associate` hard-codes `iou_batch`
even under `asso_func='giou'`, and the port preserves that asymmetry.

Note the reference's giou_batch NORMALIZES to [0, 1] (`association.py:54`,
`giou = (giou + 1.) / 2.`), so `iou_threshold = 0.3` keeps meaning "30% score" rather
than a raw GIoU of 0.3. The port does the same. (This was worth checking rather than
assuming: had the reference not normalized, every giou threshold comparison would have
meant something different and every giou association would have been wrong.)

Detections carry the same Gaussian noise as `detection-noise/` (sigma = 4 px), because
the clean regime is what hid the export-convention bug and there is no reason to trust
it here either.

WHAT THIS FIXTURE DELIBERATELY AVOIDS
-------------------------------------
Every detection here stays FAR from the image origin, and that is load-bearing.

`KalmanBoxTracker.last_observation` is initialized to the placeholder
`[-1, -1, -1, -1, -1]` and is only set on the first MATCH — so a freshly-spawned track
is "never observed". The reference's OCR stage builds its track side from
`last_boxes[unmatched_trks]` (`ocsort.py:281`), which INCLUDES those never-observed
tracks, at their placeholder box.

Under `iou` that placeholder scores 0.0000 against everything (it has zero area), so it
never matches and the question never arises — which is precisely why the default config
never exposed this. Under `giou` it scores by ENCLOSING-BOX distance, and a placeholder
sitting at (-1, -1) is close to the image origin:

    giou(placeholder, det @ [2,2,82,82])     = 0.4645   -> MATCHES (> 0.3)
    giou(placeholder, det @ [300,300,...])   = 0.0220
    giou(placeholder, det @ [600,600,...])   = 0.0069

So under giou the reference will OCR-match a never-observed track to any detection near
the origin — recovering a track from a box it never saw, using a placeholder as if it
were evidence. vestige excludes never-observed tracks from OCR (`ocsort.ts`), and does
not reproduce this. That is a DELIBERATE divergence, pinned by
`tests/unit/ocsort-ocr-sentinel.test.ts` and recorded in ADR-0007.

This fixture therefore keeps its detections away from the origin so that it tests the
giou and BYTE paths where the two implementations SHOULD agree, rather than tripping
over a known, documented disagreement. That carve-out is stated here rather than left
implicit: a fixture that quietly steers around a divergence is how the last three
green-but-wrong fixtures happened.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON are
committed together; never one without the other.

Regenerate:

    python packages/core/fixtures/ocsort-giou-byte/gen.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

OC_SORT_PATH = Path(os.environ.get("OC_SORT_PATH", str(Path.home() / "repos" / "OC_SORT")))
if not OC_SORT_PATH.exists():
    sys.exit(
        f"OC_SORT clone not found at {OC_SORT_PATH}.\n"
        f"  git clone https://github.com/noahcao/OC_SORT.git {OC_SORT_PATH}"
    )

sys.path.insert(0, str(OC_SORT_PATH))

from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402

NOAHCAO_SHA = subprocess.check_output(
    ["git", "-C", str(OC_SORT_PATH), "rev-parse", "HEAD"], text=True
).strip()

OCSORT_CONFIG = {
    "det_thresh": 0.6,
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "asso_func": "giou",  # <- the point
    "inertia": 0.2,
    "use_byte": True,  # <- the other point
}

IMG_INFO = (1080, 1920)
IMG_SIZE = (1080, 1920)

SEED = 20260714
NOISE_SIGMA = 4.0
BOX = 80.0
N_FRAMES = 60

# Score bands, per OC_SORT's update(): high = score >= det_thresh (primary stage);
# low = 0.1 < score < det_thresh (BYTE stage, only when use_byte=True); <= 0.1 dropped.
SCORE_HIGH = 0.9
SCORE_LOW = 0.4  # in (0.1, 0.6) -> BYTE stage only; cannot spawn a track

# ORIGIN_KEEPOUT: every box starts at least this far from (0, 0). See the module
# docstring — under giou the never-observed placeholder box at (-1,-1) scores 0.46
# against an origin-adjacent detection, and that is a documented divergence this
# fixture is not trying to test.
ORIGIN_KEEPOUT = 250.0

# MAKING giou ACTUALLY DISCRIMINATE
# ---------------------------------
# The first version of this sequence passed on the first run AND survived mutation:
# breaking the giou normalization, forcing the BYTE/OCR stages back to plain IoU, and
# disabling OCR entirely all left it green. It was testing almost nothing. The arithmetic
# says why, and it is worth writing down.
#
# For two BOX-sized squares offset by d with NO overlap, the reference's normalized giou
# collapses to a clean closed form:
#
#     iou            = 0                    (flat, for every d >= BOX)
#     giou_normalized = BOX / (BOX + d)     (a smooth function of separation)
#
#   d = 80  -> 0.500      d = 120 -> 0.400      d = 187 -> 0.300 (the threshold)
#   d = 100 -> 0.444      d = 160 -> 0.333      d = 200 -> 0.286
#
# So giou differs from iou EXACTLY where the boxes do not overlap: iou says "no match,"
# giou still says "close enough." Every recovery in the first sequence overlapped its
# target, which is the one regime where the two agree — so the fixture could not tell
# them apart. And OCR never fired at all: under perfectly linear motion the Kalman
# prediction stays accurate through an occlusion, so the PRIMARY stage always matched
# first and OCR was never reached.
#
# The two probes below fix that. Both use a STATIONARY track, which makes the Kalman
# prediction and the last observation coincide — so a detection displaced by DISPLACE px
# is invisible to the primary stage (which hard-codes iou_batch: 0 -> no match) while
# still being reachable by whichever stage uses asso_func:
#
#   G  BYTE-stage probe. Stationary. On frame 30 its detection is LOW-score AND displaced
#      120 px. Low score keeps it out of the primary stage entirely; the BYTE stage sees
#      it with asso_func. iou -> 0.000 (no match). giou -> 0.400 (match).
#
#   H  OCR-stage probe. Stationary, occluded for 5 frames, then reappears HIGH-score but
#      displaced 120 px. Primary fails on iou (0.000). OCR retries against the track's
#      last observation with asso_func. iou -> 0.000 (no match, so the detection spawns a
#      NEW id). giou -> 0.400 (recovered under the ORIGINAL id).
#
# Each probe therefore has a different outcome under giou than under iou, so the fixture
# now fails if the normalization is dropped ((giou+1)/2 -> raw giou turns 0.400 into
# -0.200, below threshold), if the BYTE/OCR stages ignore asso_func, or if OCR is
# disabled. Verified by mutation.
#
# DISPLACE is 120 px: comfortably clear of the 187 px point where giou itself falls below
# threshold, and comfortably clear of BOX=80 where the boxes would start to overlap and
# iou would revive. Being ~60 px from either cliff means sigma=4 noise cannot flip an
# outcome.
DISPLACE = 120.0

# name, x0, y0, vx, vy, occluded_frames, low_score_frames, displaced_frames
TRACKS_SPEC = [
    ("A", 300.0, 300.0, 3.0, 0.0, [], [], []),  # control, always high score
    ("B", 700.0, 300.0, 0.0, 2.0, [], [12, 13, 14], []),  # score dips -> BYTE stage (overlapping)
    ("C", 1100.0, 300.0, 2.0, 1.0, list(range(20, 33)), [], []),  # long occlusion
    ("D", 1600.0, 300.0, 1.0, 2.0, list(range(40, 45)), [25, 26], []),  # occlusion AND score dips
    ("G", 300.0, 700.0, 0.0, 0.0, [], [30], [30]),  # BYTE + giou discriminator
    ("H", 800.0, 700.0, 0.0, 0.0, list(range(20, 25)), [], [25]),  # OCR + giou discriminator
    ("F", 1400.0, 700.0, 0.0, -1.0, [], [50, 51, 52], []),  # late score dip
]


def true_bbox(spec, frame: int) -> list[float]:
    _, x0, y0, vx, vy, _, _, _ = spec
    return [x0 + vx * frame, y0 + vy * frame, x0 + vx * frame + BOX, y0 + vy * frame + BOX]


def build_detections(rng) -> list[list[list[float]]]:
    frames: list[list[list[float]]] = []
    for f in range(N_FRAMES):
        rows: list[list[float]] = []
        for spec in TRACKS_SPEC:
            _, _, _, _, _, occluded, low_frames, displaced_frames = spec
            if f in occluded:
                continue
            box = true_bbox(spec, f)
            if f in displaced_frames:
                box = [box[0] + DISPLACE, box[1], box[2] + DISPLACE, box[3]]
            noisy = [float(v + rng.normal(0.0, NOISE_SIGMA)) for v in box]
            if min(noisy[0], noisy[1]) < ORIGIN_KEEPOUT:
                raise AssertionError(
                    f"frame {f}: box {noisy} strays inside the origin keep-out zone; the "
                    f"giou placeholder divergence would contaminate this fixture."
                )
            score = SCORE_LOW if f in low_frames else SCORE_HIGH
            rows.append(noisy + [score])
        frames.append(rows)
    return frames


def main() -> None:
    import trackers.ocsort_tracker.ocsort as _oc

    rng = np.random.default_rng(SEED)
    frames_dets = build_detections(rng)

    _oc.KalmanBoxTracker.count = 0
    tracker = OCSort(**OCSORT_CONFIG)

    frames = []
    for f in range(N_FRAMES):
        dets_list = frames_dets[f]
        dets = np.array(dets_list, dtype=np.float64) if dets_list else np.empty((0, 5))
        res = tracker.update(dets, img_info=IMG_INFO, img_size=IMG_SIZE)
        frames.append(
            {
                "frame_index": f + 1,
                "detections": dets_list,
                "tracks_out": [[float(v) for v in row] for row in res],
            }
        )

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/ocsort-giou-byte/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "noahcao_sha": NOAHCAO_SHA,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "ocsort_config": OCSORT_CONFIG,
        "sequence": {
            "n_frames": N_FRAMES,
            "box_size": BOX,
            "noise_sigma": NOISE_SIGMA,
            "seed": SEED,
            "origin_keepout": ORIGIN_KEEPOUT,
            "score_high": SCORE_HIGH,
            "score_low": SCORE_LOW,
            "displace_px": DISPLACE,
            "tracks_spec": [
                {
                    "name": n,
                    "x0": x0,
                    "y0": y0,
                    "vx": vx,
                    "vy": vy,
                    "occluded_frames_0indexed": oc,
                    "low_score_frames_0indexed": lo,
                    "displaced_frames_0indexed": dp,
                }
                for (n, x0, y0, vx, vy, oc, lo, dp) in TRACKS_SPEC
            ],
        },
        "frames": frames,
    }

    out_path = Path(__file__).with_name("data.json")
    out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n")
    n_out = sum(len(fr["tracks_out"]) for fr in frames)
    print(
        f"wrote {out_path} ({N_FRAMES} frames, {n_out} track outputs, "
        f"asso_func=giou use_byte=True, noahcao={NOAHCAO_SHA[:8]})"
    )


if __name__ == "__main__":
    main()
