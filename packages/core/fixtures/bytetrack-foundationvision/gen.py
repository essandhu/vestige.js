"""Generate a per-frame trace of FoundationVision/ByteTrack on a synthetic sequence.

Cross-implementation oracle for
`packages/core/tests/validation/bytetrack-foundationvision-fixture.test.ts`.
The TS test runs the same sequence through `ByteTracker` and asserts that the
per-frame outputs match within tolerance.

Third and final sister fixture, after `ocsort-noahcao/` and `sort-abewley/`:
same harness shape, different reference. Retires the ByteTracker row in
`docs/decisions/0003-tracker-lifecycle-bookkeeping.md` §6.

This script imports the reference implementation from a user-provided clone of
`FoundationVision/ByteTrack` (ARCHITECTURE.md §10.2 — reference implementations
are referenced, not vendored). Set `BYTETRACK_PATH` to override the default.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON are
committed together; never one without the other.

Setup (one-time):

    git clone https://github.com/FoundationVision/ByteTrack.git ~/repos/ByteTrack
    git -C ~/repos/ByteTrack checkout d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
    cd packages/core/fixtures
    pip install -r requirements.txt   # adds lapx + cython_bbox for this fixture

Regenerate:

    python packages/core/fixtures/bytetrack-foundationvision/gen.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import scipy

DEFAULT_BYTETRACK_PATH = Path.home() / "repos" / "ByteTrack"
BYTETRACK_PATH = Path(os.environ.get("BYTETRACK_PATH", str(DEFAULT_BYTETRACK_PATH)))

if not BYTETRACK_PATH.exists():
    sys.exit(
        f"ByteTrack clone not found at {BYTETRACK_PATH}.\n"
        f"Set BYTETRACK_PATH or clone: "
        f"git clone https://github.com/FoundationVision/ByteTrack.git {BYTETRACK_PATH}"
    )

# --- Import rigging -------------------------------------------------------
#
# The reference predates numpy 2 and lives inside the YOLOX training framework.
# Three obstacles, each handled below. None of them touches tracker math.
#
# 1. `np.float` (an alias for the builtin `float`) was removed in numpy 1.24.
#    `byte_tracker.py` and `matching.py` still use it as a dtype.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

# 2. `byte_tracker.py` imports torch and `matching.py` imports cv2, but neither is
#    reachable from the code path this fixture exercises: torch is used only on the
#    non-(N, 5) detection input branch, and cv2 is never called by iou_distance /
#    fuse_score / linear_assignment. Stub both so the fixture doesn't drag in a
#    ~2 GB dependency to compute IoU on 80x80 boxes.
for _name in ("torch", "torch.nn", "torch.nn.functional", "cv2"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["torch"].nn = sys.modules["torch.nn"]  # type: ignore[attr-defined]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]  # type: ignore[attr-defined]

# 3. `yolox/__init__.py` eagerly imports the whole training framework
#    (torch.distributed, the model zoo, ...). Pre-register `yolox` and
#    `yolox.tracker` as namespace packages pointing at the real directories so
#    those __init__.py files never execute, while submodule imports
#    (`from yolox.tracker import matching`, `from .kalman_filter import ...`)
#    still resolve. Only the four files in yolox/tracker/ are ever loaded.
for _pkg_name, _pkg_dir in (
    ("yolox", BYTETRACK_PATH / "yolox"),
    ("yolox.tracker", BYTETRACK_PATH / "yolox" / "tracker"),
):
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_pkg_dir)]  # type: ignore[attr-defined]
    sys.modules[_pkg_name] = _pkg

from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker  # noqa: E402

# Lock the reference commit hash into the fixture envelope. A mismatch with the
# README's pinned hash is what tells a future reviewer the reference drifted.
FOUNDATIONVISION_SHA = subprocess.check_output(
    ["git", "-C", str(BYTETRACK_PATH), "rev-parse", "HEAD"], text=True
).strip()


class Args:
    """The subset of YOLOX's arg namespace that BYTETracker.__init__ reads.

    Values are the reference defaults (`tools/demo_track.py` / `yolox/exp` defaults):
    track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False.
    """

    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False


BYTETRACK_CONFIG = {
    "track_thresh": Args.track_thresh,
    "track_buffer": Args.track_buffer,
    "match_thresh": Args.match_thresh,
    "mot20": Args.mot20,
    "frame_rate": 30,
}

BOX_SIZE = 80.0
N_FRAMES = 90

# Score bands. `track_thresh=0.5` splits high from low; `det_thresh = 0.6` is the
# floor for spawning a new track; `0.1` is the hard-coded floor below which a
# detection is discarded entirely.
SCORE_HIGH = 0.9  # > track_thresh, and >= det_thresh -> can spawn and match in stage 1
SCORE_LOW = 0.4  # in (0.1, 0.5) -> stage-2 only; cannot spawn a track
SCORE_BELOW_FLOOR = 0.05  # <= 0.1 -> discarded entirely; the track sees a miss
SCORE_AT_THRESH = 0.5  # == track_thresh -> discarded by BOTH bands (see below)

# `max_time_lost = int(frame_rate / 30.0 * track_buffer)` = 30. A lost track is
# removed on the first frame where `frame_id - end_frame > max_time_lost`, i.e.
# after 31 consecutive misses. Tracks D and E straddle that boundary by one frame.
MAX_TIME_LOST = 30

# --- Sequence design ------------------------------------------------------
#
# Eight tracks, each in its own lane, 80x80 boxes, velocities <= 2 px/frame so no
# two tracks ever overlap and every box stays far from every association cutoff.
# That isolation is load-bearing for two reasons:
#
#   (a) `matching.ious()` goes through `cython_bbox`, which uses the Faster R-CNN
#       "+1" pixel convention ((x2-x1+1)*(y2-y1+1)); vestige's `geometry/iou.ts`
#       uses the standard convention. The two disagree by ~1-2% on partial
#       overlaps. With disjoint lanes every cost is either ~0 (a track and its own
#       detection) or 1.0 (everything else), so the discrepancy can never flip an
#       assignment. See this fixture's README for the numeric margin.
#   (b) Disjoint boxes mean no Hungarian ties, so cross-implementation id
#       assignment is deterministic.
#
# Each track targets a specific lifecycle edge:
#
#   A  control: always visible, high score. Stage-1 path, never lost.
#   B  score oscillates high/low: exercises stage-2 recovery of a CONFIRMED track.
#      Stays output every frame (the reference keeps it Tracked via stage 2).
#   C  10-frame occlusion: confirmed -> lost -> re-found via stage 1, SAME id.
#      (Re-activation timing: the frame it reappears, `re_activate` sets
#      is_activated=True, so it IS output that frame.)
#   D  EXACTLY 30-frame occlusion: at the last missed frame tsu == 30, which is
#      NOT > max_time_lost, so it survives and re-associates with its SAME id.
#      This is the "survives at exactly maxAge" side of the off-by-one.
#   E  32-frame occlusion: tsu passes 31 > 30, the track is REMOVED, and the
#      reappearing detection spawns a NEW id. The other side of the off-by-one.
#      D and E bracket the boundary -- that pair is the sharpest test here.
#
#      Why 32 and not 31: the reference has a one-frame RESURRECTION WINDOW. In
#      byte_tracker.py:283-284 the bookkeeping runs
#
#          self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
#          self.removed_stracks.extend(removed_stracks)
#
#      -- the subtraction uses the ACCUMULATED removed list, which does not yet
#      contain the tracks removed on THIS frame; they are appended on the next
#      line. So a just-removed track survives in `lost_stracks` for exactly one
#      more frame, stays in that frame's `strack_pool`, and if its detection
#      reappears right then, stage 1 `re_activate()`s it under its ORIGINAL id.
#      (The object ends up in `tracked_stracks` and `removed_stracks` at once.)
#
#      A 31-frame occlusion lands squarely in that window and the reference keeps
#      the old id, which is NOT the max_time_lost semantics this pair is meant to
#      test. vestige deliberately does not reproduce the resurrection (see this
#      fixture's README and ADR-0003); the sequence therefore steps one frame
#      clear of the window so that D/E test the boundary itself rather than the
#      bug. `tests/unit/bytetrack-resurrection.test.ts` pins vestige's behavior
#      inside the window explicitly.
#   F  late spawn (frame 5): tentative -> confirmed via a STAGE-3 match on its
#      second frame. Not output on its spawn frame.
#   G  the one-chance rule for tentative tracks, made OBSERVABLE. G is static and
#      appears on frame 12, spawning a tentative track that is never output. It is
#      then absent for frames 13..39, so on frame 13 -- its second frame -- it goes
#      unmatched in stage 3 and the reference `mark_removed()`s it immediately (a
#      tentative gets exactly one chance; it is NOT retained for max_time_lost).
#      On frame 40 a detection reappears at the SAME position.
#
#      The reappearance is what gives the rule teeth. A tentative track is never
#      exported, so a tracker that simply FAILED to remove G would look identical
#      for frames 13..39 -- the bug would be invisible. But on frame 40:
#         correct  -> G is long gone; the detection spawns a NEW id (tentative),
#                     which is confirmed via stage 3 on frame 41 and output from 41.
#         buggy    -> a lingering tentative G still sits at that exact bbox, wins the
#                     stage-3 match, and is confirmed under its ORIGINAL id -- output
#                     one frame EARLY (frame 40) and with the WRONG id.
#      Both the frame index and the id differ, so the fixture catches it twice over.
#      (Verified by mutation: deleting the `state = 'removed'` line in bytetrack.ts
#      step 6 makes this test fail. Without G's reappearance it did not.)
#   H  score-boundary probe on an established track:
#        frame 30 -> score 0.05 (<= 0.1 floor)      -> detection DISCARDED -> miss -> lost
#        frame 31 -> score 0.9                       -> re-found via stage 1, SAME id
#        frame 50 -> score 0.5 (== track_thresh)     -> detection DISCARDED -> miss -> lost
#        frame 51 -> score 0.9                       -> re-found via stage 1, SAME id
#      The frame-50 case is the subtle one: the reference's bands are
#      `remain = scores > track_thresh` and `second = (scores > 0.1) & (scores < track_thresh)`,
#      so a detection scoring EXACTLY track_thresh falls into neither band and is
#      silently dropped. vestige documents that quirk (bytetrack.ts ByteTrackerOptions);
#      this frame is what proves it actually reproduces it.
#
# (name, x0, y0, vx, vy, first_frame, last_frame, occluded_frames, score_overrides)
TRACKS_SPEC = [
    ("A", 50.0, 50.0, 2.0, 0.0, 0, 89, [], {}),
    ("B", 400.0, 50.0, 0.0, 2.0, 0, 89, [], {f: SCORE_LOW for f in (10, 11, 25, 40, 41, 42)}),
    ("C", 750.0, 50.0, 2.0, 0.0, 0, 89, list(range(20, 30)), {}),
    ("D", 50.0, 400.0, 0.0, 2.0, 0, 89, list(range(26, 56)), {}),
    ("E", 400.0, 400.0, 1.0, 1.0, 0, 89, list(range(26, 58)), {}),
    ("F", 750.0, 400.0, 2.0, 0.0, 5, 89, [], {}),
    ("G", 50.0, 750.0, 0.0, 0.0, 12, 89, list(range(13, 40)), {}),
    (
        "H",
        400.0,
        750.0,
        2.0,
        0.0,
        0,
        89,
        [],
        {30: SCORE_BELOW_FLOOR, 50: SCORE_AT_THRESH},
    ),
]


def true_bbox(spec, frame: int) -> list[float]:
    """Ground-truth [x1, y1, x2, y2] for a track at a frame.

    Position is measured from the track's own first frame, so a late-spawning
    track (F) starts at its declared (x0, y0) rather than where it would have
    been had it existed since frame 0.
    """
    _, x0, y0, vx, vy, first, _, _, _ = spec
    dt = frame - first
    x = x0 + vx * dt
    y = y0 + vy * dt
    return [x, y, x + BOX_SIZE, y + BOX_SIZE]


def build_detections(frame: int) -> list[list[float]]:
    """Per-frame [[x1, y1, x2, y2, score], ...] honoring visibility and score overrides."""
    rows: list[list[float]] = []
    for spec in TRACKS_SPEC:
        _, _, _, _, _, first, last, occluded, score_overrides = spec
        if frame < first or frame > last:
            continue
        if frame in occluded:
            continue
        score = score_overrides.get(frame, SCORE_HIGH)
        rows.append(true_bbox(spec, frame) + [score])
    return rows


def main() -> None:
    # BaseTrack._count is a CLASS-level counter that survives across BYTETracker
    # instances; reset it so the fixture's track ids start at 1 deterministically.
    BaseTrack._count = 0

    tracker = BYTETracker(Args(), frame_rate=BYTETRACK_CONFIG["frame_rate"])

    # `BYTETracker.update` rescales boxes by
    # `scale = min(img_size[0] / img_h, img_size[1] / img_w)`. Passing img_info ==
    # img_size makes scale exactly 1.0, so the synthetic boxes pass through
    # untouched and the fixture stays in detection coordinates.
    img_info = [1080, 1920]
    img_size = [1080, 1920]

    frames: list[dict] = []
    for f in range(N_FRAMES):
        dets_list = build_detections(f)
        # Fresh array each frame: `update` does an in-place `bboxes /= scale` on a
        # view of this array. (scale == 1.0 here, so it is a numeric no-op, but
        # relying on that would be fragile.)
        dets = (
            np.array(dets_list, dtype=np.float64)
            if dets_list
            else np.empty((0, 5), dtype=np.float64)
        )
        output_stracks = tracker.update(dets, img_info, img_size)
        # The reference returns STrack objects. `STrack.tlbr` is [x1, y1, x2, y2],
        # derived from the current Kalman mean -- the same quantity vestige's
        # `Track.bbox` holds.
        out_rows = [
            [
                float(t.tlbr[0]),
                float(t.tlbr[1]),
                float(t.tlbr[2]),
                float(t.tlbr[3]),
                int(t.track_id),
            ]
            for t in output_stracks
        ]
        frames.append(
            {
                "frame_index": f + 1,
                "detections": dets_list,
                "tracks_out": out_rows,
            }
        )

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/bytetrack-foundationvision/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "foundationvision_sha": FOUNDATIONVISION_SHA,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "bytetrack_config": BYTETRACK_CONFIG,
        "sequence": {
            "n_frames": N_FRAMES,
            "box_size": BOX_SIZE,
            "max_time_lost": MAX_TIME_LOST,
            "tracks_spec": [
                {
                    "name": name,
                    "x0": x0,
                    "y0": y0,
                    "vx": vx,
                    "vy": vy,
                    "first_frame_0indexed": first,
                    "last_frame_0indexed": last,
                    "occluded_frames_0indexed": occluded,
                    "score_overrides_0indexed": {str(k): v for k, v in overrides.items()},
                }
                for (name, x0, y0, vx, vy, first, last, occluded, overrides) in TRACKS_SPEC
            ],
        },
        "frames": frames,
    }

    out_path = Path(__file__).with_name("data.json")
    out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n")
    n_out = sum(len(fr["tracks_out"]) for fr in frames)
    print(
        f"wrote {out_path} ({N_FRAMES} frames, {n_out} track outputs, "
        f"foundationvision={FOUNDATIONVISION_SHA[:8]})"
    )


if __name__ == "__main__":
    main()
