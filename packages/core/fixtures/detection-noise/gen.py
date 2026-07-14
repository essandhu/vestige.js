"""Drive all three references through a sequence with REALISTIC DETECTION NOISE.

Cross-implementation oracle for
`packages/core/tests/validation/detection-noise-fixture.test.ts`.

Why this fixture exists
-----------------------
Every synthetic fixture in this repo feeds the trackers NOISE-FREE detections on
perfectly linear trajectories. That is not a neutral simplification -- it is a regime,
and it hides at least one whole class of bug.

Under noise-free linear motion, a Kalman filter's posterior converges exactly onto the
measurement: the two become numerically identical (measured delta: 0.000000 px). So a
tracker that exports the KALMAN STATE and one that exports the RAW DETECTION produce
byte-identical output, and no fixture built on clean detections can tell them apart.

They are not the same thing. `noahcao/OC_SORT` exports the raw last observation
(`ocsort.py:313-320`):

    if trk.last_observation.sum() < 0:
        d = trk.get_state()[0]        # never observed -> Kalman state
    else:
        d = trk.last_observation[:4]  # <- the RAW DETECTION

vestige exported the Kalman posterior, with a code comment asserting the two were
"equal-by-design to the last observation within Kalman-filter noise". They are equal
only in the regime the fixtures happened to use. On any real detector stream they
diverge -- which is the entire point of running a Kalman filter at all.

(`sort.py` exports `get_state()` and ByteTrack exports `STrack.tlbr`, both
Kalman-derived, so the port is correct for those two. They are driven here anyway:
proving the noise regime does NOT break them is what isolates the OC-SORT bug, and a
fixture that only ever drives the tracker you already suspect is not evidence.)

The noise is Gaussian, sigma = 4 px per box coordinate -- a modest, realistic
localization error for a detector on 80x80 boxes. It is applied to the DETECTIONS only;
ground-truth motion stays linear, so any divergence is about how the trackers handle
measurement noise, not about an exotic trajectory.

Determinism: the noise is drawn once here from a seeded RNG and the resulting noisy
boxes are COMMITTED in `data.json`. The TS test replays those exact boxes, so nothing
depends on reproducing numpy's RNG in JavaScript.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON are
committed together; never one without the other.

Setup: all three reference clones (see `association-crowded/gen.py`).

Regenerate:

    python packages/core/fixtures/detection-noise/gen.py
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

SORT_PATH = Path(os.environ.get("SORT_PATH", str(Path.home() / "repos" / "sort")))
OC_SORT_PATH = Path(os.environ.get("OC_SORT_PATH", str(Path.home() / "repos" / "OC_SORT")))
BYTETRACK_PATH = Path(os.environ.get("BYTETRACK_PATH", str(Path.home() / "repos" / "ByteTrack")))

for _p, _name, _url in (
    (SORT_PATH, "sort", "https://github.com/abewley/sort.git"),
    (OC_SORT_PATH, "OC_SORT", "https://github.com/noahcao/OC_SORT.git"),
    (BYTETRACK_PATH, "ByteTrack", "https://github.com/FoundationVision/ByteTrack.git"),
):
    if not _p.exists():
        sys.exit(f"{_name} clone not found at {_p}.\n  git clone {_url} {_p}")

# --- Import rigging (identical to association-crowded/gen.py; see there for the why) ---
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

_mpl = types.ModuleType("matplotlib")
_mpl.use = lambda *_a, **_kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("matplotlib", _mpl)
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("matplotlib.patches", types.ModuleType("matplotlib.patches"))
_skimage = types.ModuleType("skimage")
_skimage_io = types.ModuleType("skimage.io")
_skimage.io = _skimage_io  # type: ignore[attr-defined]
sys.modules.setdefault("skimage", _skimage)
sys.modules.setdefault("skimage.io", _skimage_io)

for _n in ("torch", "torch.nn", "torch.nn.functional", "cv2"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["torch"].nn = sys.modules["torch.nn"]  # type: ignore[attr-defined]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]  # type: ignore[attr-defined]
sys.modules["torch"].Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
for _pkg_name, _pkg_dir in (
    ("yolox", BYTETRACK_PATH / "yolox"),
    ("yolox.tracker", BYTETRACK_PATH / "yolox" / "tracker"),
):
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_pkg_dir)]  # type: ignore[attr-defined]
    sys.modules[_pkg_name] = _pkg

sys.path.insert(0, str(SORT_PATH))
sys.path.insert(0, str(OC_SORT_PATH))

from sort import KalmanBoxTracker, Sort  # noqa: E402
from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402
from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker  # noqa: E402


def _sha(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


SORT_CONFIG = {"max_age": 5, "min_hits": 3, "iou_threshold": 0.3}
OCSORT_CONFIG = {
    "det_thresh": 0.6,
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "asso_func": "iou",
    "inertia": 0.2,
    "use_byte": False,
}


class ByteArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False


BYTETRACK_CONFIG = {
    "track_thresh": ByteArgs.track_thresh,
    "track_buffer": ByteArgs.track_buffer,
    "match_thresh": ByteArgs.match_thresh,
    "mot20": ByteArgs.mot20,
    "frame_rate": 30,
}

IMG_INFO = (1080, 1920)
IMG_SIZE = (1080, 1920)

SEED = 20260713
NOISE_SIGMA = 4.0  # px, per box coordinate — a modest, realistic detector localization error
BOX = 80.0
SCORE = 0.9
N_FRAMES = 60

# Well-separated lanes: the point of THIS fixture is measurement noise, not crowding
# (that is `association-crowded/`). Keeping the boxes far apart means any divergence
# here is attributable to noise handling and nothing else. Occlusions are included so
# the noisy regime is exercised through re-association too — OC-SORT's ORU replays a
# virtual trajectory between the pre- and post-occlusion OBSERVATIONS, so noise on
# those two boxes is exactly what it is sensitive to.
#
# name, x0, y0, vx, vy, occluded_frames
TRACKS_SPEC = [
    ("A", 100.0, 100.0, 3.0, 0.0, []),  # control: always visible
    ("B", 500.0, 100.0, 0.0, 3.0, []),  # control: always visible
    ("C", 900.0, 100.0, 2.0, 1.0, list(range(20, 32))),  # 12-frame occlusion -> ORU + OCR
    ("D", 100.0, 500.0, 1.0, 2.0, list(range(35, 41))),  # 6-frame occlusion
    ("E", 500.0, 500.0, -2.0, 1.0, [15, 16]),  # brief 2-frame occlusion
    ("F", 900.0, 500.0, 0.0, -1.0, []),  # control
]


def true_bbox(spec, frame: int) -> list[float]:
    _, x0, y0, vx, vy, _ = spec
    x = x0 + vx * frame
    y = y0 + vy * frame
    return [x, y, x + BOX, y + BOX]


def build_detections(rng) -> list[list[list[float]]]:
    """Per-frame [[x1, y1, x2, y2, score], ...] with independent Gaussian noise on
    every coordinate of every detection."""
    frames: list[list[list[float]]] = []
    for f in range(N_FRAMES):
        rows: list[list[float]] = []
        for spec in TRACKS_SPEC:
            if f in spec[5]:
                continue
            box = true_bbox(spec, f)
            noisy = [float(v + rng.normal(0.0, NOISE_SIGMA)) for v in box]
            # Guard the degenerate case: noise must never invert a box (x2 < x1). At
            # sigma=4 on an 80px box this cannot happen in practice, but a fixture that
            # silently emits an inverted box would be testing the wrong thing.
            if noisy[2] <= noisy[0] or noisy[3] <= noisy[1]:
                raise AssertionError(f"noise inverted a box at frame {f}: {noisy}")
            rows.append(noisy + [SCORE])
        frames.append(rows)
    return frames


def run_sort(frames_dets):
    KalmanBoxTracker.count = 0
    t = Sort(**SORT_CONFIG)
    out = []
    for dets_list in frames_dets:
        dets = np.array(dets_list, dtype=np.float64) if dets_list else np.empty((0, 5))
        res = t.update(dets)
        out.append([[float(v) for v in row] for row in res])
    return out


def run_ocsort(frames_dets):
    import trackers.ocsort_tracker.ocsort as _oc

    _oc.KalmanBoxTracker.count = 0
    t = OCSort(**OCSORT_CONFIG)
    out = []
    for dets_list in frames_dets:
        dets = np.array(dets_list, dtype=np.float64) if dets_list else np.empty((0, 5))
        res = t.update(dets, img_info=IMG_INFO, img_size=IMG_SIZE)
        out.append([[float(v) for v in row] for row in res])
    return out


def run_bytetrack(frames_dets):
    BaseTrack._count = 0
    t = BYTETracker(ByteArgs(), frame_rate=BYTETRACK_CONFIG["frame_rate"])
    out = []
    for dets_list in frames_dets:
        dets = np.array(dets_list, dtype=np.float64) if dets_list else np.empty((0, 5))
        res = t.update(dets, IMG_INFO, IMG_SIZE)
        out.append(
            [
                [float(s.tlbr[0]), float(s.tlbr[1]), float(s.tlbr[2]), float(s.tlbr[3]), int(s.track_id)]
                for s in res
            ]
        )
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    frames_dets = build_detections(rng)

    sort_out = run_sort(frames_dets)
    ocsort_out = run_ocsort(frames_dets)
    byte_out = run_bytetrack(frames_dets)

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/detection-noise/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "abewley_sha": _sha(SORT_PATH),
            "noahcao_sha": _sha(OC_SORT_PATH),
            "foundationvision_sha": _sha(BYTETRACK_PATH),
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "sort_config": SORT_CONFIG,
        "ocsort_config": OCSORT_CONFIG,
        "bytetrack_config": BYTETRACK_CONFIG,
        "sequence": {
            "n_frames": N_FRAMES,
            "box_size": BOX,
            "noise_sigma": NOISE_SIGMA,
            "seed": SEED,
            "tracks_spec": [
                {"name": n, "x0": x0, "y0": y0, "vx": vx, "vy": vy, "occluded_frames_0indexed": oc}
                for (n, x0, y0, vx, vy, oc) in TRACKS_SPEC
            ],
        },
        "frames": [
            {
                "frame_index": f + 1,
                "detections": frames_dets[f],
                "sort_out": sort_out[f],
                "ocsort_out": ocsort_out[f],
                "bytetrack_out": byte_out[f],
            }
            for f in range(N_FRAMES)
        ],
    }

    out_path = Path(__file__).with_name("data.json")
    out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {out_path} ({N_FRAMES} frames, sigma={NOISE_SIGMA}px; "
        f"sort={sum(len(r) for r in sort_out)} ocsort={sum(len(r) for r in ocsort_out)} "
        f"bytetrack={sum(len(r) for r in byte_out)} track outputs)"
    )


if __name__ == "__main__":
    main()
