# Generate the TrackEval cross-implementation fixture for the MOT17 loader +
# preprocessing + metrics pipeline (see README.md in this directory).
#
# Requires: numpy==1.23.5 scipy==1.10.1 (see ../requirements.txt) and a clone
# of JonathonLuiten/TrackEval at the commit pinned in the README, importable
# via PYTHONPATH or TRACKEVAL_PATH below.
#
# Usage:
#   TRACKEVAL_PATH=/path/to/TrackEval python3 gen.py
#
# The script is fully deterministic: the synthetic sequence is hand-specified
# (no RNG), so regeneration only changes the JSON if TrackEval or the pinned
# numeric stack changes.

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

TRACKEVAL_PATH = os.environ.get('TRACKEVAL_PATH', '/tmp/TrackEval')
sys.path.insert(0, TRACKEVAL_PATH)

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import trackeval  # noqa: E402

SEQ = 'SYN-01'
NUM_FRAMES = 12


def gt_lines():
    """Synthetic MOT17-style gt covering every preprocessing branch.

    - id 1: pedestrian (class 1, considered)      frames 1-12, drifting right.
    - id 2: pedestrian (class 1, considered)      frames 1-8, drifting left.
    - id 3: static_person (class 7, considered)   frames 1-12 -> distractor:
            tracker dets matched to it must be REMOVED by preprocessing.
    - id 4: pedestrian but zero-marked (consider 0) frames 1-12 -> dropped from
            eval gt; tracker dets matched to it are NOT removed (-> FPs).
    - id 5: car (class 3, considered)             frames 3-9 -> valid non-eval
            class: dropped from eval gt, does NOT remove tracker dets (-> FPs).
    """
    lines = []
    for t in range(1, NUM_FRAMES + 1):
        lines.append(f'{t},1,{100 + 6 * t},200,60,180,1,1,1')
        if t <= 8:
            lines.append(f'{t},2,{400 - 4 * t},300,55,170,1,1,1')
        lines.append(f'{t},3,800,400,50,160,1,7,1')
        lines.append(f'{t},4,1200,500,60,180,0,1,0.6')
        if 3 <= t <= 9:
            lines.append(f'{t},5,1500,600,120,80,1,3,1')
    return lines


def tracker_lines():
    """Synthetic tracker output.

    - id 101 follows gt 1 (offset +3,+2) frames 1-12.
    - id 102 follows gt 2 frames 1-5; id 103 takes over frames 6-8 (id switch).
    - id 104 follows the distractor gt 3 frames 1-10 -> removed by preproc.
    - id 105 follows zero-marked gt 4 frames 2-7 -> kept -> 6 FPs.
    - id 106 is a far spurious box frames 4-5 -> 2 FPs.
    - id 107 follows the car gt 5 frames 3-6 -> kept -> 4 FPs.
    """
    lines = []
    for t in range(1, NUM_FRAMES + 1):
        lines.append(f'{t},101,{103 + 6 * t},202,60,180,0.95,-1,-1,-1')
        if t <= 5:
            lines.append(f'{t},102,{402 - 4 * t},303,55,170,0.9,-1,-1,-1')
        if 6 <= t <= 8:
            lines.append(f'{t},103,{402 - 4 * t},303,55,170,0.88,-1,-1,-1')
        if t <= 10:
            lines.append(f'{t},104,802,402,50,160,0.85,-1,-1,-1')
        if 2 <= t <= 7:
            lines.append(f'{t},105,1203,502,60,180,0.8,-1,-1,-1')
        if 4 <= t <= 5:
            lines.append(f'{t},106,50,800,40,100,0.7,-1,-1,-1')
        if 3 <= t <= 6:
            lines.append(f'{t},107,1502,602,120,80,0.75,-1,-1,-1')
    return lines


def main():
    work = tempfile.mkdtemp(prefix='trackeval-fixture-')
    gt_dir = os.path.join(work, 'gt', 'MOT17-train', SEQ, 'gt')
    tr_dir = os.path.join(work, 'trackers', 'MOT17-train', 'vestige', 'data')
    os.makedirs(gt_dir)
    os.makedirs(tr_dir)
    with open(os.path.join(gt_dir, 'gt.txt'), 'w') as f:
        f.write('\n'.join(gt_lines()) + '\n')
    with open(os.path.join(tr_dir, SEQ + '.txt'), 'w') as f:
        f.write('\n'.join(tracker_lines()) + '\n')

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        'GT_FOLDER': os.path.join(work, 'gt'),
        'TRACKERS_FOLDER': os.path.join(work, 'trackers'),
        'OUTPUT_FOLDER': os.path.join(work, 'out'),
        'TRACKERS_TO_EVAL': ['vestige'],
        'BENCHMARK': 'MOT17',
        'SPLIT_TO_EVAL': 'train',
        'SEQ_INFO': {SEQ: NUM_FRAMES},
        'DO_PREPROC': True,
        'PRINT_CONFIG': False,
    })
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        'PRINT_RESULTS': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
        'OUTPUT_EMPTY_CLASSES': False,
        'LOG_ON_ERROR': None,
    })

    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator = trackeval.Evaluator(eval_config)
    output_res, _ = evaluator.evaluate([dataset], metrics)
    res = output_res['MotChallenge2DBox']['vestige'][SEQ]['pedestrian']

    # Per-frame preprocessed ids. TrackEval relabels kept ids to be contiguous
    # in ascending original-id order; the fixture stores the relabeled arrays
    # and the TS test applies the same ascending relabeling to its own output.
    raw = dataset.get_raw_seq_data('vestige', SEQ)
    pre = dataset.get_preprocessed_seq_data(raw, 'pedestrian')
    frames = [
        {
            'gtIds': [int(x) for x in pre['gt_ids'][t]],
            'trackerIds': [int(x) for x in pre['tracker_ids'][t]],
        }
        for t in range(NUM_FRAMES)
    ]

    def scalars(family, fields):
        out = {}
        for f in fields:
            v = res[family][f]
            if isinstance(v, np.ndarray):
                out[f] = {'perAlpha': [float(x) for x in v], 'mean': float(np.mean(v))}
            else:
                out[f] = float(v)
        return out

    trackeval_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=TRACKEVAL_PATH, text=True).strip()

    fixture = {
        '$schema': 'vestige.js fixture v1',
        'generator': {
            'script': 'packages/eval/fixtures/trackeval-preproc/gen.py',
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'trackeval': trackeval_commit,
            'generated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        'sequence': {
            'name': SEQ,
            'numFrames': NUM_FRAMES,
            'gtLines': gt_lines(),
            'trackerLines': tracker_lines(),
        },
        'preprocessed': {
            'numGtIds': int(pre['num_gt_ids']),
            'numTrackerIds': int(pre['num_tracker_ids']),
            'numGtDets': int(pre['num_gt_dets']),
            'numTrackerDets': int(pre['num_tracker_dets']),
            'frames': frames,
        },
        'metrics': {
            'hota': scalars('HOTA', ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']),
            'clear': scalars('CLEAR', ['MOTA', 'MOTP', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']),
            'identity': scalars('Identity', ['IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP']),
        },
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.json')
    with open(out_path, 'w') as f:
        json.dump(fixture, f, indent=2)
        f.write('\n')
    print('wrote', out_path)


if __name__ == '__main__':
    main()
