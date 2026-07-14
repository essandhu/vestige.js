"""Generate TrackEval oracles for HOTA / CLEAR-MOT / Identity.

Cross-implementation oracle for
`packages/eval/tests/validation/trackeval-metrics-fixture.test.ts`.

Why this fixture exists
-----------------------
Until it landed, the metric implementations in `packages/eval/src/metrics/` had NO
cross-implementation oracle. Their unit tests state their expected values were
"hand-traced from the HOTA definition ... as implemented by TrackEval metrics/hota.py"
-- i.e. hand-computed toy cases, not generated from TrackEval.

That is the wrong shape of assurance for this code. These metrics are the instrument
we measure the trackers WITH. A subtle error in HOTA's AssA denominators or in
CLEAR's ID-switch bookkeeping does not announce itself: every published number is
quietly wrong, and it looks like a TRACKER bug. The instrument has to be calibrated
against the canonical implementation, not against our own reading of the paper.

Two live bugs were found by exactly this comparison and are pinned by the scenarios
below (see ADR-0006):

  * `clearmot.ts` committed its continuity memory OUTSIDE the "did this frame match
    anything" guard, so any EMPTY frame wiped it. TrackEval `clear.py:70-76` hits
    `continue` on an empty timestep, BEFORE the reset -- so its memory survives the
    hole. Result: spurious ID switches and spurious Frags after any dropout. Pinned
    by `empty-tracker-frame`, `empty-gt-frame`, `fragmentation`.
  * `solveLsap` broke assignment TIES the opposite way from scipy, which is what
    TrackEval matches with. Pinned by `tied-boxes`.

Scope: this fixture drives the METRICS, not the loader or the preprocessing (which
has its own fixture, `trackeval-preproc/`). It builds TrackEval's `data` dict
directly and calls `eval_sequence` on it, so nothing here depends on file I/O.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON are
committed together; never one without the other.

Setup (one-time):

    git clone https://github.com/JonathonLuiten/TrackEval.git ~/repos/TrackEval
    cd packages/eval/fixtures && pip install -r requirements.txt

Regenerate:

    python packages/eval/fixtures/trackeval-metrics/gen.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_TRACKEVAL_PATH = Path.home() / "repos" / "TrackEval"
TRACKEVAL_PATH = Path(os.environ.get("TRACKEVAL_PATH", str(DEFAULT_TRACKEVAL_PATH)))

if not TRACKEVAL_PATH.exists():
    sys.exit(
        f"TrackEval clone not found at {TRACKEVAL_PATH}.\n"
        f"Set TRACKEVAL_PATH or clone: "
        f"git clone https://github.com/JonathonLuiten/TrackEval.git {TRACKEVAL_PATH}"
    )

# TrackEval predates numpy 2, which removed the `np.int` / `np.float` / `np.bool`
# aliases for the builtins. `identity.py` and `clear.py` still cast with them. Restore
# the aliases before importing; they are exact synonyms for the builtins, so this
# changes no arithmetic (the same shim the bytetrack fixture needs for `np.float`).
for _alias, _builtin in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)

sys.path.insert(0, str(TRACKEVAL_PATH))

from trackeval.datasets._base_dataset import _BaseDataset  # noqa: E402
from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402

TRACKEVAL_SHA = subprocess.check_output(
    ["git", "-C", str(TRACKEVAL_PATH), "rev-parse", "HEAD"], text=True
).strip()

THRESHOLD = 0.5  # the standard CLEAR / Identity similarity threshold

# --- Scenarios -------------------------------------------------------------
#
# Each scenario is a list of frames; each frame is (gt, tracker) where each side is a
# list of (id, [x1, y1, x2, y2]). Boxes are xyxy here (vestige's convention) and are
# converted to xywh for TrackEval, which is what its IoU expects.
#
# Every scenario targets a specific way a metric can be wrong.

B = lambda x: [float(x), 100.0, float(x) + 60.0, 160.0]  # noqa: E731  60x60 box at height 100


def _shift(x0: float, vx: float, t: int) -> list[float]:
    return B(x0 + vx * t)


def perfect() -> list[tuple[list, list]]:
    """Two objects tracked flawlessly. Every metric must be 1.0. The control."""
    frames = []
    for t in range(10):
        gt = [(1, _shift(100, 5, t)), (2, _shift(400, -3, t))]
        tr = [(101, _shift(100, 5, t)), (102, _shift(400, -3, t))]
        frames.append((gt, tr))
    return frames


def id_switch() -> list[tuple[list, list]]:
    """Tracker swaps the two ids halfway. Pins IDSW, IDF1's trajectory matching, AssA."""
    frames = []
    for t in range(10):
        gt = [(1, _shift(100, 5, t)), (2, _shift(400, -3, t))]
        a, b = (101, 102) if t < 5 else (102, 101)
        tr = [(a, _shift(100, 5, t)), (b, _shift(400, -3, t))]
        frames.append((gt, tr))
    return frames


def empty_tracker_frame() -> list[tuple[list, list]]:
    """The detector drops out entirely on frames 4-5, then resumes with the SAME id.

    THE continuity-memory case. TrackEval `continue`s past an empty timestep without
    resetting `prev_timestep_tracker_id`, so on resume the gt is still considered
    "previously tracked": no ID switch, no new fragment. A metric that blanks its
    memory on the hole invents both.
    """
    frames = []
    for t in range(10):
        gt = [(1, _shift(100, 5, t)), (2, _shift(400, -3, t))]
        tr = [] if t in (4, 5) else [(101, _shift(100, 5, t)), (102, _shift(400, -3, t))]
        frames.append((gt, tr))
    return frames


def empty_gt_frame() -> list[tuple[list, list]]:
    """No ground truth at all on frames 3-4 (every gt filtered as a distractor, say),
    while the tracker keeps reporting. Those tracker boxes are pure FP, and the
    continuity memory must again survive the hole."""
    frames = []
    for t in range(10):
        gt = [] if t in (3, 4) else [(1, _shift(100, 5, t)), (2, _shift(400, -3, t))]
        tr = [(101, _shift(100, 5, t)), (102, _shift(400, -3, t))]
        frames.append((gt, tr))
    return frames


def fragmentation() -> list[tuple[list, list]]:
    """One object is lost for 3 frames and recovered under the SAME id; the other is
    continuous. Pins Frag and MT/PT/ML."""
    frames = []
    for t in range(12):
        gt = [(1, _shift(100, 5, t)), (2, _shift(400, -3, t))]
        tr = [(102, _shift(400, -3, t))]
        if t not in (4, 5, 6):
            tr.insert(0, (101, _shift(100, 5, t)))
        frames.append((gt, tr))
    return frames


def tied_boxes() -> list[tuple[list, list]]:
    """A single gt with TWO tracker boxes sitting exactly on it — a genuine assignment
    tie — where the two tied choices have DIFFERENT FUTURES.

    This is the scenario that pins the solver's tie-break rule, and getting it to
    discriminate takes care. Simply duplicating boxes does NOT work: totals like IDTP
    and CLR_TP are invariant under a tie (both choices are equally good by definition),
    so MOTA and IDF1 come out the same either way and the test proves nothing.

    The trick is to make the two tied candidates diverge AFTER the tie is resolved:

        gt 1      present frames 0-9
        tracker 101   on the gt box, frames 0-4 ONLY   <- appears first, so dense id 0
        tracker 102   on the gt box, frames 0-9

    On frame 0 both score exactly 1.0 against gt 1 and neither has a continuity bonus
    yet, so the assignment is a pure tie and the solver's rule alone decides it. From
    then on the +1000 continuity bonus locks in whichever it chose:

      * choose 101 (scipy's choice): it vanishes after frame 4, so gt 1 is handed to
        102 on frame 5 -> an ID SWITCH, and MOTA drops accordingly.
      * choose 102: gt 1 is tracked cleanly for all 10 frames -> NO id switch.

    Same TP, same FP, different IDSW -> different MOTA. That is the observable the tie
    rule moves, and nothing else in either package exercises it: every tracker fixture
    uses well-separated boxes, so no two IoUs are ever exactly equal.
    """
    frames = []
    for t in range(10):
        box = _shift(200, 2, t)
        gt = [(1, list(box))]
        tr = [(102, list(box))]
        if t <= 4:
            tr.insert(0, (101, list(box)))  # first-seen -> dense tracker id 0
        frames.append((gt, tr))
    return frames


def fp_fn_mix() -> list[tuple[list, list]]:
    """A spurious tracker box (FP) and a missed gt (FN) alongside a good match."""
    frames = []
    for t in range(10):
        gt = [(1, _shift(100, 5, t)), (2, _shift(400, -3, t))]
        tr = [(101, _shift(100, 5, t))]
        if t % 3 == 0:
            tr.append((103, _shift(700, 0, t)))  # FP: matches no gt at all
        frames.append((gt, tr))  # gt id 2 is never tracked -> pure FN / ML
    return frames


def truncated_tracker() -> list[tuple[list, list]]:
    """The tracker run stops early while ground truth continues. The trailing gt must
    still be counted as FN."""
    frames = []
    for t in range(12):
        gt = [(1, _shift(100, 5, t)), (2, _shift(400, -3, t))]
        tr = [] if t >= 9 else [(101, _shift(100, 5, t)), (102, _shift(400, -3, t))]
        frames.append((gt, tr))
    return frames


def near_threshold() -> list[tuple[list, list]]:
    """Tracker boxes drift so IoU sweeps across the 0.5 threshold, and HOTA's alpha
    sweep sees a wide spread of similarities. Pins the threshold comparison direction
    and the alpha integral."""
    frames = []
    for t in range(12):
        gt = [(1, B(100))]
        # offset grows 0,4,8,... -> IoU falls (60-off)/(60+off): 1.0, .875, .765, ...
        tr = [(101, B(100 + 4 * t))]
        frames.append((gt, tr))
    return frames


SCENARIOS = {
    "perfect": perfect,
    "id-switch": id_switch,
    "empty-tracker-frame": empty_tracker_frame,
    "empty-gt-frame": empty_gt_frame,
    "fragmentation": fragmentation,
    "tied-boxes": tied_boxes,
    "fp-fn-mix": fp_fn_mix,
    "truncated-tracker": truncated_tracker,
    "near-threshold": near_threshold,
}


def to_trackeval_data(frames: list[tuple[list, list]]) -> dict:
    """Build TrackEval's preprocessed `data` dict directly (ids densified 0-based,
    similarity computed with TrackEval's OWN IoU so the comparison isolates the
    metric logic rather than re-testing IoU)."""
    gt_id_map: dict[int, int] = {}
    tr_id_map: dict[int, int] = {}
    for gt, tr in frames:
        for i, _ in gt:
            gt_id_map.setdefault(i, len(gt_id_map))
        for i, _ in tr:
            tr_id_map.setdefault(i, len(tr_id_map))

    gt_ids, tracker_ids, sims = [], [], []
    num_gt_dets = num_tracker_dets = 0
    for gt, tr in frames:
        g_ids = np.array([gt_id_map[i] for i, _ in gt], dtype=int)
        t_ids = np.array([tr_id_map[i] for i, _ in tr], dtype=int)
        num_gt_dets += len(g_ids)
        num_tracker_dets += len(t_ids)

        if len(gt) and len(tr):
            # xyxy -> xywh, which is what TrackEval's box IoU expects.
            g_box = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in gt], dtype=float)
            t_box = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in tr], dtype=float)
            sim = _BaseDataset._calculate_box_ious(g_box, t_box, box_format="xywh")
        else:
            sim = np.zeros((len(gt), len(tr)))

        gt_ids.append(g_ids)
        tracker_ids.append(t_ids)
        sims.append(sim)

    return {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tr_id_map),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": sims,
    }


def _f(v) -> float:
    return float(np.asarray(v).item()) if np.asarray(v).size == 1 else float(np.mean(v))


def main() -> None:
    clear = CLEAR({"THRESHOLD": THRESHOLD, "PRINT_CONFIG": False})
    ident = Identity({"THRESHOLD": THRESHOLD, "PRINT_CONFIG": False})
    hota = HOTA({"PRINT_CONFIG": False})

    out_scenarios = []
    raw_clear: dict = {}
    raw_ident: dict = {}
    raw_hota: dict = {}
    for name, build in SCENARIOS.items():
        frames = build()
        data = to_trackeval_data(frames)

        c = clear.eval_sequence(data)
        i = ident.eval_sequence(data)
        h = hota.eval_sequence(data)
        raw_clear[name] = c
        raw_ident[name] = i
        raw_hota[name] = h

        out_scenarios.append(
            {
                "name": name,
                "frames": [
                    {
                        "gt": [{"id": gid, "bbox": box} for gid, box in gt],
                        "track": [{"id": tid, "bbox": box} for tid, box in tr],
                    }
                    for gt, tr in frames
                ],
                "expected": {
                    "clear": {
                        "tp": _f(c["CLR_TP"]),
                        "fp": _f(c["CLR_FP"]),
                        "fn": _f(c["CLR_FN"]),
                        "idsw": _f(c["IDSW"]),
                        "mota": _f(c["MOTA"]),
                        "motp": _f(c["MOTP"]),
                        "mt": _f(c["MT"]),
                        "pt": _f(c["PT"]),
                        "ml": _f(c["ML"]),
                        "frag": _f(c["Frag"]),
                    },
                    "identity": {
                        "idtp": _f(i["IDTP"]),
                        "idfp": _f(i["IDFP"]),
                        "idfn": _f(i["IDFN"]),
                        "idf1": _f(i["IDF1"]),
                        "idp": _f(i["IDP"]),
                        "idr": _f(i["IDR"]),
                    },
                    "hota": {
                        "hota": _f(np.mean(h["HOTA"])),
                        "deta": _f(np.mean(h["DetA"])),
                        "assa": _f(np.mean(h["AssA"])),
                        "loca": _f(np.mean(h["LocA"])),
                        "hota_per_alpha": [float(v) for v in h["HOTA"]],
                        "deta_per_alpha": [float(v) for v in h["DetA"]],
                        "assa_per_alpha": [float(v) for v in h["AssA"]],
                        "loca_per_alpha": [float(v) for v in h["LocA"]],
                    },
                },
            }
        )

    # THE AGGREGATION ORACLE.
    #
    # Treat the nine scenarios above as nine SEQUENCES of one benchmark, and ask TrackEval
    # to combine them with its OWN combine_sequences -- the same code path that turns
    # MOT17's seven sequences into the single published HOTA / MOTA / IDF1.
    #
    # This is emphatically not a mean of the per-sequence numbers. CLEAR sums the counts
    # (including MOTP_sum) and re-derives; HOTA sums TP/FN/FP per alpha but TP-WEIGHTS
    # AssA/AssRe/AssPr/LocA and then recomputes HOTA = sqrt(DetA * AssA). The naive average
    # is emitted alongside, purely so the fixture records HOW DIFFERENT it is -- see the
    # validation test, which asserts the two do not coincide (otherwise the scenarios would
    # not be discriminating and the test would pass for a wrong implementation).
    comb_clear = clear.combine_sequences(raw_clear)
    comb_ident = ident.combine_sequences(raw_ident)
    comb_hota = hota.combine_sequences(raw_hota)

    naive_mota = float(np.mean([_f(raw_clear[k]["MOTA"]) for k in raw_clear]))
    naive_idf1 = float(np.mean([_f(raw_ident[k]["IDF1"]) for k in raw_ident]))
    naive_hota = float(np.mean([float(np.mean(raw_hota[k]["HOTA"])) for k in raw_hota]))

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/eval/fixtures/trackeval-metrics/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "trackeval_sha": TRACKEVAL_SHA,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "threshold": THRESHOLD,
        "scenarios": out_scenarios,
        "combined": {
            "note": (
                "TrackEval's own combine_sequences over all scenarios above, treated as "
                "sequences of one benchmark. This is the shape of the number MOTChallenge "
                "publishes. `naive_mean_*` is what you get by averaging the per-sequence "
                "values instead -- recorded to show it is a DIFFERENT number, not to bless it."
            ),
            "clear": {
                "tp": _f(comb_clear["CLR_TP"]),
                "fp": _f(comb_clear["CLR_FP"]),
                "fn": _f(comb_clear["CLR_FN"]),
                "idsw": _f(comb_clear["IDSW"]),
                "mota": _f(comb_clear["MOTA"]),
                "motp": _f(comb_clear["MOTP"]),
                "mt": _f(comb_clear["MT"]),
                "pt": _f(comb_clear["PT"]),
                "ml": _f(comb_clear["ML"]),
                "frag": _f(comb_clear["Frag"]),
            },
            "identity": {
                "idtp": _f(comb_ident["IDTP"]),
                "idfp": _f(comb_ident["IDFP"]),
                "idfn": _f(comb_ident["IDFN"]),
                "idf1": _f(comb_ident["IDF1"]),
                "idp": _f(comb_ident["IDP"]),
                "idr": _f(comb_ident["IDR"]),
            },
            "hota": {
                "hota": float(np.mean(comb_hota["HOTA"])),
                "deta": float(np.mean(comb_hota["DetA"])),
                "assa": float(np.mean(comb_hota["AssA"])),
                "loca": float(np.mean(comb_hota["LocA"])),
                "hota_per_alpha": [float(v) for v in comb_hota["HOTA"]],
                "deta_per_alpha": [float(v) for v in comb_hota["DetA"]],
                "assa_per_alpha": [float(v) for v in comb_hota["AssA"]],
                "loca_per_alpha": [float(v) for v in comb_hota["LocA"]],
            },
            "naive_mean_mota": naive_mota,
            "naive_mean_idf1": naive_idf1,
            "naive_mean_hota": naive_hota,
        },
    }

    out_path = Path(__file__).with_name("data.json")
    out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_path} ({len(out_scenarios)} scenarios, trackeval={TRACKEVAL_SHA[:8]})")
    for s in out_scenarios:
        e = s["expected"]
        print(
            f"  {s['name']:22s} MOTA={e['clear']['mota']:7.4f} IDF1={e['identity']['idf1']:7.4f} "
            f"HOTA={e['hota']['hota']:7.4f} IDSW={e['clear']['idsw']:.0f} Frag={e['clear']['frag']:.0f}"
        )


if __name__ == "__main__":
    main()
