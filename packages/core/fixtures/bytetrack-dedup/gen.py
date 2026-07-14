"""Drive FoundationVision/ByteTrack through a sequence that TRIGGERS remove_duplicate_stracks.

Cross-implementation oracle for
`packages/core/tests/validation/bytetrack-dedup-fixture.test.ts`.

Why this fixture exists
-----------------------
`remove_duplicate_stracks` (byte_tracker.py:317-330) fires only when a TRACKED track and
a LOST track overlap at IoU > 0.85. No fixture in the repo has ever put two tracks in
that position -- every sequence keeps its boxes well separated precisely so ids stay
unambiguous. So the function shipped with zero reference-backed coverage, and it turned
out to be wrong in a way that is not subtle.

The reference ranks the two candidates by "how long has this track been alive AND
tracked":

    timep = stracksa[p].frame_id - stracksa[p].start_frame   # a = TRACKED
    timeq = stracksb[q].frame_id - stracksb[q].start_frame   # b = LOST
    if timep > timeq: drop the lost one
    else:             drop the tracked one

The trap is `frame_id`. It is written ONLY in activate() / re_activate() / update(), so
for a LOST track it FREEZES at the frame it was last matched. `timeq` therefore excludes
the lost span -- it is the track's lifetime *as of when it went lost*, not as of now.
vestige compared `age`, which keeps incrementing every frame including while lost, so the
two quantities drift apart by exactly `timeSinceUpdate` and the comparison inverts.

The consequence is not a cosmetic id difference. With two tracks spawned on the same
frame, `age` is EQUAL for both, `c.age > l.age` is false, and vestige deletes the LIVE,
JUST-MATCHED track in favour of the stale lost one. Executed on the sequence below:

    frame 4:  reference -> [(1, A)]              vestige -> []           <- live track deleted
    frame 5:  reference -> [(1, A)]              vestige -> [(2, B)]     <- wrong box
    frame 6:  reference -> [(1, A), (3, B)]      vestige -> [(2, B), (3, A)]   <- A permanently id 3

The correct port-side expression is `age - timeSinceUpdate`, on BOTH sides, which
reconstructs the reference's frozen `frame_id - start_frame` exactly.

Sequence design
---------------
Dedup only fires on a >0.85 overlap, so this fixture deliberately does what every other
one avoids: it puts two boxes almost on top of each other (100x100, offset 4 px ->
IoU 0.923) and then removes one detection, so the survivor is TRACKED and the vanished
one is LOST while their boxes still coincide.

Boxes are STATIONARY. That is load-bearing: a lost track's Kalman prediction would
otherwise drift away from its live twin and the overlap would fall back under 0.85 before
the bug could show. Stationary boxes keep the pair locked together for as long as the
lost track survives.

Both members of each pair spawn on the SAME frame. That is the discriminating condition —
it makes `age` equal for both, which is exactly where comparing `age` instead of
`age - timeSinceUpdate` flips the answer. A pair with different spawn ages would mask the
bug, because the raw ages would already be ordered correctly.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON are
committed together; never one without the other.

Regenerate:

    python packages/core/fixtures/bytetrack-dedup/gen.py
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

BYTETRACK_PATH = Path(os.environ.get("BYTETRACK_PATH", str(Path.home() / "repos" / "ByteTrack")))
if not BYTETRACK_PATH.exists():
    sys.exit(
        f"ByteTrack clone not found at {BYTETRACK_PATH}.\n"
        f"  git clone https://github.com/FoundationVision/ByteTrack.git {BYTETRACK_PATH}"
    )

# Import rigging — see bytetrack-foundationvision/gen.py for the full story.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
for _n in ("torch", "torch.nn", "torch.nn.functional", "cv2"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["torch"].nn = sys.modules["torch.nn"]  # type: ignore[attr-defined]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]  # type: ignore[attr-defined]
for _pkg_name, _pkg_dir in (
    ("yolox", BYTETRACK_PATH / "yolox"),
    ("yolox.tracker", BYTETRACK_PATH / "yolox" / "tracker"),
):
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_pkg_dir)]  # type: ignore[attr-defined]
    sys.modules[_pkg_name] = _pkg

from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker  # noqa: E402

FOUNDATIONVISION_SHA = subprocess.check_output(
    ["git", "-C", str(BYTETRACK_PATH), "rev-parse", "HEAD"], text=True
).strip()


class Args:
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

IMG_INFO = (1080, 1920)
IMG_SIZE = (1080, 1920)

SCORE = 0.9
BOX = 100.0
# 4 px offset on a 100 px box: intersection 96*100, union 10000+10000-9600 = 10400
# -> IoU = 0.923, comfortably above the 0.85 dedup cutoff (pdist < 0.15) and far enough
# above it that Kalman jitter cannot drop the pair below the line.
TWIN_OFFSET = 4.0
N_FRAMES = 40

# Three stationary near-duplicate PAIRS. Both members of a pair spawn on frame 0, so
# their ages are identical -- the condition under which comparing `age` instead of
# `age - timeSinceUpdate` inverts the reference's ranking.
#
# In each pair exactly one member's detection disappears for a stretch, leaving it LOST
# while its twin stays TRACKED and their boxes still coincide. That is the only shape
# that reaches remove_duplicate_stracks at all.
#
# (name, x, y, occluded_frames)
TRACKS_SPEC = [
    # Pair 1: the TRAILING twin vanishes.
    ("P1_lead", 100.0, 100.0, []),
    ("P1_trail", 100.0 + TWIN_OFFSET, 100.0, list(range(8, 14))),
    # Pair 2: the LEADING twin vanishes — same shape, opposite member, so a fix that
    # happens to favour one side of the pair does not pass by luck.
    ("P2_lead", 500.0, 100.0, list(range(20, 26))),
    ("P2_trail", 500.0 + TWIN_OFFSET, 100.0, []),
    # Pair 3: a longer dropout, to keep the lost twin around while `age` and
    # `age - timeSinceUpdate` diverge further (tsu grows to 10).
    ("P3_lead", 900.0, 100.0, []),
    ("P3_trail", 900.0 + TWIN_OFFSET, 100.0, list(range(28, 38))),
    # Control: isolated and always visible. Must keep one id in both implementations.
    ("SOLO", 1400.0, 400.0, []),
]


def build_detections(frame: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for _name, x, y, occluded in TRACKS_SPEC:
        if frame in occluded:
            continue
        rows.append([x, y, x + BOX, y + BOX, SCORE])
    return rows


def main() -> None:
    BaseTrack._count = 0
    tracker = BYTETracker(Args(), frame_rate=BYTETRACK_CONFIG["frame_rate"])

    frames = []
    for f in range(N_FRAMES):
        dets_list = build_detections(f)
        dets = np.array(dets_list, dtype=np.float64) if dets_list else np.empty((0, 5))
        out = tracker.update(dets, IMG_INFO, IMG_SIZE)
        frames.append(
            {
                "frame_index": f + 1,
                "detections": dets_list,
                "tracks_out": [
                    [float(s.tlbr[0]), float(s.tlbr[1]), float(s.tlbr[2]), float(s.tlbr[3]), int(s.track_id)]
                    for s in out
                ],
            }
        )

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/bytetrack-dedup/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "foundationvision_sha": FOUNDATIONVISION_SHA,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "bytetrack_config": BYTETRACK_CONFIG,
        "sequence": {
            "n_frames": N_FRAMES,
            "box_size": BOX,
            "twin_offset": TWIN_OFFSET,
            "twin_iou": 0.923,
            "tracks_spec": [
                {"name": n, "x": x, "y": y, "occluded_frames_0indexed": oc}
                for (n, x, y, oc) in TRACKS_SPEC
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
    ids = sorted({row[4] for fr in frames for row in fr["tracks_out"]})
    print(f"  reference emitted ids: {ids}")


if __name__ == "__main__":
    main()
