# vestige.js — Architecture Document

A zero-dependency, TypeScript-first multi-object tracking library implementing the production tracking-by-detection family (SORT, ByteTrack, OC-SORT, BoT-SORT) for browser and Node runtimes.

> *A vestige is a persistent trace of something across time — exactly what a tracked object is. The library keeps identities through frames of motion, occlusion, and change.*

---

## Table of contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Design principles](#2-design-principles)
3. [Project structure](#3-project-structure)
4. [Core data model](#4-core-data-model)
5. [Numerical core (zero dependencies)](#5-numerical-core-zero-dependencies)
6. [Tracker architecture](#6-tracker-architecture)
7. [Algorithm offerings — research-backed scope](#7-algorithm-offerings--research-backed-scope)
8. [Plugin interfaces — extension points for learned and classical components](#8-plugin-interfaces--extension-points-for-learned-and-classical-components)
9. [Performance strategy](#9-performance-strategy)
10. [Accuracy strategy and benchmarking](#10-accuracy-strategy-and-benchmarking)
11. [Public API surface](#11-public-api-surface)
12. [Build, packaging, and distribution](#12-build-packaging-and-distribution)
13. [Testing strategy](#13-testing-strategy)
14. [Contributor model](#14-contributor-model)
15. [Roadmap and milestones](#15-roadmap-and-milestones)
16. [Risks and open questions](#16-risks-and-open-questions)
17. [Reference appendix](#17-reference-appendix)
18. [Naming](#18-naming)

---

## 1. Goals and non-goals

### 1.1 Goals

- **Zero runtime dependencies.** Every algorithmic component implemented from first principles. The only allowed `dependencies` field in `package.json` is empty.
- **Modern tracking-by-detection family.** Implementations of SORT, ByteTrack, OC-SORT, and BoT-SORT, each faithful to its published reference implementation and paper.
- **Production-grade accuracy.** HOTA, MOTA, and IDF1 numbers on MOT17 and MOT20 within a small, documented delta of the published Python reference numbers.
- **Production-grade performance.** Throughput targets in the 1,000+ tracks-per-second range on a single thread for moderate scene complexity (10–50 tracks per frame). No claims of beating Python+C; instead, transparent benchmarks against realistic JS baselines.
- **Browser and Node parity.** Identical behavior across runtimes. No DOM dependencies. No `fs` dependencies in core. Bundles cleanly under any common bundler.
- **Detector-agnostic.** Any source of bounding-box detections works (Transformers.js, ONNX Runtime Web, MediaPipe, TensorFlow.js, remote API, mocked data, file-loaded ground truth).
- **Inspectable state.** Track lifecycle is fully observable. No hidden state machines.

### 1.2 Non-goals

- **Detection.** Users bring their own detector. No bundled model, no inference runtime.
- **Streaming framework.** The user owns the frame loop. The library is pull-based, not push-based.
- **UI / visualization.** No canvas drawing, no overlay rendering, no React components.
- **Transformer-based end-to-end trackers** (MOTIP, SambaMOTR, CO-MOT). Different architecture, different runtime needs, different project.
- **Mobile-specific bindings.** React Native may work but is not a supported target initially.
- **Training.** This library does inference-time tracking only. Embedding models, if used, are provided by the user.

---

## 2. Design principles

### 2.1 Pure functional core, imperative shell

The mathematical components — Kalman filter prediction, IoU computation, Hungarian assignment, cost matrix construction — are pure functions over plain data. The tracker classes are the thin imperative shell that holds frame-to-frame state. This separation enables exhaustive unit testing of the algorithmic core without instantiating a tracker.

### 2.2 Algorithms own their options

There is no shared `TrackerOptions` union to maintain. Each tracker (`SortTracker`, `ByteTracker`, `OcSortTracker`, `BotSortTracker`) has its own options interface, derived from its paper's hyperparameters. This avoids the coupling that traps API designers when new variants are added.

### 2.3 Pull, not push

The user calls `tracker.update(detections)` once per frame. The library does not own a frame loop, a video element, or a worker. Users compose the library with whatever frame source they prefer.

### 2.4 Typed contracts, behavioral interfaces

The data flowing through the tracker is strictly typed (`Detection`, `Track`, `TrackState`). The behavioral extension points — motion predictors, cost functions, association strategies, Re-ID embedders, CMC providers — are interfaces with structural typing. Classical implementations (Kalman filter, IoU, Hungarian) ship as defaults; learned alternatives (LSTM motion predictors, fused IoU+Re-ID costs, neural association) can be plugged in without modifying the core. This pattern preserves customization without trapping users in literal unions, and positions the library to absorb learned components as the research matures.

### 2.5 Numerical determinism where possible

Track ID assignment, tie-breaking in Hungarian assignment, and lifecycle ordering are deterministic given the same input sequence. Non-determinism (e.g. from set/map iteration order) is explicitly forbidden in core paths. This is what makes regression-test snapshots viable.

### 2.6 Explicit cost over magic ergonomics

When a design choice trades clarity for cleverness, choose clarity. A verbose API that an interviewer can read top-to-bottom in an evening is the goal. Method chaining, implicit conversions, and "magic" defaults are avoided.

---

## 3. Project structure

```
vestige.js/
├── packages/
│   ├── core/                       # Zero-dependency core
│   │   ├── src/
│   │   │   ├── types.ts            # Detection, Track, TrackState
│   │   │   ├── geometry/
│   │   │   │   ├── bbox.ts         # xyxy/xywh/cxcywh conversions
│   │   │   │   ├── iou.ts          # IoU, GIoU, DIoU, CIoU
│   │   │   │   └── linalg.ts       # 8x8 matrix ops, Cholesky, etc.
│   │   │   ├── solvers/
│   │   │   │   └── hungarian.ts    # Jonker-Volgenant or Munkres
│   │   │   ├── filters/
│   │   │   │   ├── kalman.ts       # Generic linear KF
│   │   │   │   └── motion-models/
│   │   │   │       ├── cv-bbox.ts  # SORT-style state vector
│   │   │   │       └── cv-xyah.ts  # DeepSORT-style state vector
│   │   │   ├── trackers/
│   │   │   │   ├── base.ts         # Shared Track lifecycle
│   │   │   │   ├── sort.ts         # SortTracker (sync)
│   │   │   │   ├── bytetrack.ts    # ByteTracker (sync)
│   │   │   │   ├── ocsort.ts       # OcSortTracker (sync)
│   │   │   │   ├── botsort.ts      # BotSortTracker (sync)
│   │   │   │   └── async/          # Async variants (v0.3+)
│   │   │   │       ├── bytetrack.ts    # AsyncByteTracker
│   │   │   │       ├── ocsort.ts       # AsyncOcSortTracker
│   │   │   │       └── botsort.ts      # AsyncBotSortTracker
│   │   │   ├── plugins/
│   │   │   │   ├── motion.ts       # MotionPredictor interface + Kalman defaults
│   │   │   │   ├── cost.ts         # CostFunction interface + IoU/GIoU defaults
│   │   │   │   ├── association.ts  # AssociationStrategy interface + Hungarian default
│   │   │   │   ├── reid.ts         # Embedder interface (no default)
│   │   │   │   └── cmc.ts          # CmcProvider interface + SparseOpticalFlowCmc
│   │   │   └── index.ts            # Public exports
│   │   └── tests/
│   │       ├── unit/               # Pure-function tests
│   │       ├── property/           # Invariant tests
│   │       └── snapshot/           # Deterministic regression
│   └── eval/                       # Evaluation harness (separate package)
│       └── src/
│           ├── motchallenge/       # MOT17/MOT20 loaders
│           ├── metrics/
│           │   ├── hota.ts
│           │   ├── clearmot.ts     # MOTA, MOTP
│           │   └── identity.ts     # IDF1
│           └── runner.ts
├── docs/
├── benchmarks/
└── examples/
    ├── browser-yolov8/             # Transformers.js + ByteTracker
    ├── node-cli/                   # File-based detections
    ├── learned-motion-demo/        # Hypothetical learned MotionPredictor
    └── benchmark-vs-python/
```

The split between `core/` and `eval/` is deliberate. The core package has zero dependencies and ships to end users. The eval package can take Node-only dev dependencies (e.g. file I/O for MOTChallenge sequence loading) without polluting the runtime bundle.

---

## 4. Core data model

### 4.1 Detection

```ts
interface Detection<TPayload = unknown> {
  /** Bounding box in [x1, y1, x2, y2] (top-left, bottom-right) pixel coordinates. */
  bbox: readonly [number, number, number, number];
  /** Detector confidence in [0, 1]. */
  score: number;
  /** Optional class identifier. Trackers can be class-aware or class-agnostic. */
  classId?: number;
  /** User-defined payload preserved through the tracker. */
  payload?: TPayload;
}
```

Notes:

- **xyxy is the canonical input format.** Trackers may internally convert to xyah (DeepSORT-style) or cxcywh (BoT-SORT-style); conversions are pure functions in `geometry/bbox.ts`.
- **`readonly` tuple type** prevents accidental in-place mutation of detector outputs.
- **Generic `TPayload`** lets users carry through embeddings, masks, or arbitrary metadata without the library knowing about them.

### 4.2 Track

```ts
interface Track<TPayload = unknown> extends Detection<TPayload> {
  /** Stable identifier assigned by the tracker. */
  id: number;
  /** Total frames the track has existed (including unmatched). */
  age: number;
  /** Total frames the track has been successfully matched to a detection. */
  hits: number;
  /** Consecutive frames since last successful match (0 means matched this frame). */
  timeSinceUpdate: number;
  /** Current lifecycle phase. */
  state: TrackState;
}

type TrackState = 'tentative' | 'confirmed' | 'lost' | 'removed';
```

Notes:

- **Lifecycle states** match the standard tracking-by-detection convention. `tentative` tracks have insufficient hits to be reported externally. `confirmed` tracks are publicly exposed. `lost` tracks remain in memory for potential re-association. `removed` tracks are scheduled for cleanup.
- **`Track extends Detection`** means consumers iterate one shape, not two. The tracker's job is to add an `id` and lifecycle info to a detection.

### 4.3 Tracker interface

```ts
interface Tracker<TPayload = unknown> {
  /** Process one frame of detections; returns currently-confirmed tracks. */
  update(detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[];

  /** Inspect internal state without modifying it. */
  getActiveTracks(): Track<TPayload>[];
  getLostTracks(): Track<TPayload>[];

  /** Reset all internal state. */
  reset(): void;

  /** Current frame counter; advances by 1 per update() call. */
  readonly frameIndex: number;
}
```

The interface is identical across all four trackers. The constructor options differ.

---

## 5. Numerical core (zero dependencies)

The library lives or dies by the correctness of these primitives. They are implemented from scratch, exhaustively tested, and small enough to audit.

### 5.1 Bounding box geometry

Pure functions over plain arrays. All conversions are between `xyxy` (input/output canonical), `xywh`, `cxcywh`, and `xyah` (aspect-ratio form used by DeepSORT and OC-SORT).

```ts
// geometry/bbox.ts
export function xyxyToXyah(b: BBox): [number, number, number, number] {
  const [x1, y1, x2, y2] = b;
  const w = x2 - x1;
  const h = y2 - y1;
  return [x1 + w / 2, y1 + h / 2, w / h, h];
}
```

### 5.2 IoU variants

- **IoU** — standard intersection-over-union; foundational for all trackers.
- **GIoU** — Generalized IoU (Rezatofighi et al., CVPR 2019). Used by OC-SORT as an alternative cost metric.
- **DIoU / CIoU** — Distance-IoU and Complete-IoU (Zheng et al., AAAI 2020). Available as opt-in cost functions.

Each function takes two `BBox`es (or two `BBox[]`s for batched form) and returns a scalar (or matrix) in `[-1, 1]` (GIoU) or `[0, 1]` (IoU). Batched forms operate on `Float64Array` row-major matrices for cache locality.

### 5.3 Linear algebra primitives

The Kalman filter requires:

- 8×8 matrix multiplication (state covariance updates)
- 4×8 matrix multiplication (measurement updates)
- 4×4 matrix inversion (innovation covariance — done via Cholesky decomposition for numerical stability, not naive inversion)
- Matrix-vector multiplication
- Outer product

These are implemented as fixed-size routines with unrolled loops. Total LOC under 200. Generic linear algebra is intentionally avoided — fixed sizes are dramatically faster and simpler to test.

**Cholesky decomposition is the key correctness choice.** Naive matrix inversion of the innovation covariance accumulates floating-point error, especially when detections are tightly clustered. Cholesky-based solve produces the same algebraic result with significantly better numerical conditioning. This matches what scipy uses under the hood for `numpy.linalg.solve`.

### 5.4 Hungarian / linear sum assignment

**Algorithm choice: Jonker-Volgenant over Munkres.** The Jonker-Volgenant algorithm (Jonker & Volgenant, *Computing* 1987) is asymptotically O(n³) like Munkres but in practice runs 2-10x faster on sparse rectangular cost matrices typical of tracking (typically more detections than tracks or vice versa). scipy's `linear_sum_assignment` switched to a Jonker-Volgenant variant in 2019 for the same reason.

Implementation reference: scipy's `_lsap.cpp` is the canonical reference. The algorithm is ~300 lines of dense but well-documented code. Implementing this from scratch is the single largest algorithmic investment in the project. It is also the strongest "I actually understand assignment problems" signal in the codebase.

Tie-breaking: deterministic by row-then-column index. This matters for snapshot tests.

Inputs marked as forbidden (e.g. cost ≥ threshold) are handled by setting `Number.POSITIVE_INFINITY` in the cost matrix; the algorithm respects this without special-casing.

### 5.5 Kalman filter

A generic linear Kalman filter with pluggable motion models:

```ts
// filters/kalman.ts
export class KalmanFilter<TStateDim extends number, TMeasDim extends number> {
  constructor(public readonly model: MotionModel<TStateDim, TMeasDim>) {}

  predict(state: KalmanState<TStateDim>): KalmanState<TStateDim>;
  update(state: KalmanState<TStateDim>, measurement: Float64Array): KalmanState<TStateDim>;
}
```

The two motion models initially supported:

- **`cv-bbox`** — SORT-style 7-dimensional state `[u, v, s, r, u̇, v̇, ṡ]` where `u, v` are center coordinates, `s` is scale (area), and `r` is aspect ratio (constant). This is what the original SORT paper uses.
- **`cv-xyah`** — DeepSORT/ByteTrack/OC-SORT style 8-dimensional state `[u, v, a, h, u̇, v̇, ȧ, ḣ]` where `a` is aspect ratio and `h` is height, both with velocity components.

The motion model exposes the state transition matrix F, measurement matrix H, process noise Q, and measurement noise R. Crucially, Q and R are functions of state, not constants — DeepSORT scales them by the bounding box height, and this is faithfully reproduced.

The Kalman filter is not Extended or Unscented. Linear KF is sufficient because the constant-velocity assumption holds for human-scale motion at typical frame rates. OC-SORT's contribution is not a non-linear filter; it's a smarter handling of the linear filter during occlusion.

### 5.6 Cost matrix construction

A pure function: given a set of predicted track bboxes, a set of detections, and a cost function (IoU, GIoU, etc.), produce an `M × N` cost matrix as a `Float64Array`. This is the input to Hungarian assignment.

Cost matrix gating: distances above a threshold are set to `+Infinity` before solving. This is faster than post-filtering and lets the solver short-circuit forbidden cells.

---

## 6. Tracker architecture

### 6.1 Shared lifecycle (BaseTracker)

All four trackers share a common Track lifecycle abstraction:

```ts
abstract class BaseTracker<TPayload> {
  protected tracks = new Map<number, InternalTrack<TPayload>>();
  protected nextId = 1;
  protected frameIndex = 0;

  abstract associate(detections: Detection<TPayload>[]): AssociationResult;

  update(detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[] {
    this.frameIndex++;
    this.predictAll();
    const { matched, unmatchedDets, unmatchedTracks } = this.associate([...detections]);
    this.updateMatched(matched);
    this.markUnmatchedTracks(unmatchedTracks);
    this.spawnNewTracks(unmatchedDets);
    this.advanceLifecycle();
    return this.exportConfirmed();
  }
}
```

The key extension point is `associate()`. Every tracker variant differs primarily in how association is performed:

- **SORT**: single-pass IoU + Hungarian
- **ByteTrack**: two-pass (high-score IoU, then low-score IoU on unmatched tracks)
- **OC-SORT**: ByteTrack-style two-pass + observation re-update (ORU) + observation-centric momentum (OCM)
- **BoT-SORT**: ByteTrack association + GMC-corrected predictions + optional appearance cost

Lifecycle parameters (min hits for confirmation, max age before removal, etc.) are shared but configurable per tracker.

### 6.2 SortTracker

The simplest tracker. Single-pass association with IoU cost and Hungarian assignment. Serves as:

1. **Baseline** for benchmarking improvements.
2. **Pedagogical reference** for understanding the family.
3. **Validation target** — the original SORT implementation is small enough that bit-equivalence is a realistic goal.

Hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `maxAge` | 1 | Frames a track survives without a match. Original SORT uses 1 (no occlusion handling). |
| `minHits` | 3 | Hits required before track is confirmed. |
| `iouThreshold` | 0.3 | IoU below this is treated as no-match. |

### 6.3 ByteTracker

The first real improvement. Implements the two-stage association of ByteTrack: high-confidence detections associate first, then unmatched tracks attempt to match low-confidence detections (which would have been discarded by SORT).

Hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `trackThresh` | 0.5 | Detections above this go into stage 1. |
| `trackBuffer` | 30 | Frames a lost track is retained for re-association. |
| `matchThresh` | 0.8 | IoU threshold (as 1 - cost) for valid matches. |
| `frameRate` | 30 | Used to scale `trackBuffer` if exposed in time units. |

### 6.4 OcSortTracker

Adds three components on top of SORT to address Kalman filter drift during occlusion:

- **Observation-centric Re-update (ORU)** — when a lost track is re-associated, re-run the Kalman update over a virtual trajectory between the last observation and the new one, correcting the accumulated drift.
- **Observation-centric Momentum (OCM)** — incorporate the direction of motion (computed from observations, not predictions) into the cost function.
- **Observation-centric Recovery (OCR)** — second-pass matching using last-observed positions to recover tracks that drifted out of IoU range.

Hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `detThresh` | 0.6 | Detection score threshold. |
| `maxAge` | 30 | Lost track retention. |
| `minHits` | 3 | Confirmation threshold. |
| `iouThreshold` | 0.3 | First-pass IoU threshold. |
| `deltaT` | 3 | Frames over which OCM direction is computed. |
| `asoFunc` | `'iou'` | Association cost function (`'iou'` or `'giou'`). |
| `inertia` | 0.2 | OCM weight in the association cost. |

### 6.5 BotSortTracker

Adds:

- **Camera Motion Compensation (CMC)** — global motion estimate is applied to predicted track positions before association, to handle moving cameras. Original paper uses ECC image registration; the library exposes a `CmcProvider` interface and ships a sparse-optical-flow implementation.
- **Modified Kalman filter state vector** — `[x, y, w, h, ẋ, ẏ, ẇ, ḣ]` directly, instead of the aspect-ratio-and-height form. Tracks aspect ratio changes more accurately.
- **Optional appearance cost** — when an `Embedder` is plugged in, appearance similarity (cosine distance over normalized embeddings) is fused with IoU cost via the IoU-distance and ReID-distance combination from the paper.

Hyperparameters: superset of ByteTracker plus:

| Parameter | Default | Description |
|---|---|---|
| `cmc` | `null` | Optional `CmcProvider`. |
| `embedder` | `null` | Optional `Embedder`. |
| `appearanceThresh` | 0.25 | Cosine distance threshold. |
| `proximityThresh` | 0.5 | IoU threshold gating appearance fusion. |
| `withReid` | `false` | Enables appearance branch. |

---

## 7. Algorithm offerings — research-backed scope

Each algorithm has a canonical paper, an official implementation, and reported benchmark numbers. This is the canon to implement against.

### 7.1 SORT (Simple Online and Realtime Tracking)

- **Paper**: Bewley, Ge, Ott, Ramos, Upcroft. "Simple Online and Realtime Tracking." ICIP 2016. arXiv:1602.00763.
- **Official implementation**: https://github.com/abewley/sort (Python, GPL).
- **Reported performance**: ~260 Hz on a single CPU thread; 33.4 MOTA on MOT15 with FrRCNN detections.
- **What to read**: The paper is short (4 pages) and the reference implementation is ~400 lines. Both are mandatory reading. Pay particular attention to the use of the area-and-aspect-ratio state vector and the assumption of constant aspect ratio.

### 7.2 ByteTrack (Multi-Object Tracking by Associating Every Detection Box)

- **Paper**: Zhang, Sun, Jiang, Yu, Weng, Yuan, Luo, Liu, Wang. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV 2022. arXiv:2110.06864.
- **Official implementation**: https://github.com/FoundationVision/ByteTrack (Python, MIT).
- **Reported performance**: 80.3 MOTA, 77.3 IDF1, 63.1 HOTA on MOT17 test set. 30 FPS end-to-end with YOLOX-X detector.
- **What to read**: The paper is the canonical reference for the two-stage association idea. Section 3.2 ("BYTE") describes the algorithm in pseudo-code; this is what to translate. The Kalman filter implementation is inherited from ByteTrack's predecessor and uses the DeepSORT-style state vector.
- **Companion**: ByteTrackV2 (arXiv:2303.15334) extends this to 3D tracking; not implementing this, but the 2D portion of v2 is the cleanest writeup of the association logic.

### 7.3 OC-SORT (Observation-Centric SORT)

- **Paper**: Cao, Pang, Weng, Khirodkar, Kitani. "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking." CVPR 2023. arXiv:2203.14360.
- **Official implementation**: https://github.com/noahcao/OC_SORT (Python, MIT).
- **Reported performance**: 63.2 HOTA on MOT17, 62.1 on MOT20, 55.1 on DanceTrack. 700+ FPS on a single CPU.
- **What to read**: Sections 3.2 (ORU), 3.3 (OCM), and 3.4 (OCR) define the three contributions. The official implementation is the cleanest reference for ORU specifically — the virtual trajectory math is subtle.
- **Why include this**: ByteTrack and SORT both fail when a track is lost for many frames during occlusion (Kalman drift compounds). OC-SORT specifically targets this failure mode. It's particularly important for DanceTrack-style scenarios with heavy non-linear motion.

### 7.4 BoT-SORT (Robust Associations Multi-Pedestrian Tracking)

- **Paper**: Aharon, Orfaig, Bobrovsky. "BoT-SORT: Robust Associations Multi-Pedestrian Tracking." arXiv:2206.14651, 2022.
- **Official implementation**: https://github.com/NirAharon/BoT-SORT (Python, MIT).
- **Reported performance**: 80.5 MOTA, 80.2 IDF1, 65.0 HOTA on MOT17 (with ReID). 80.2 MOTA, 77.5 IDF1, 63.3 HOTA on MOT17 without ReID.
- **What to read**: Sections 3.1 (KF state vector), 3.2 (CMC), and 3.3 (IoU-ReID fusion) define the three contributions. The CMC implementation in the official repo uses OpenCV's `findTransformECC` — this is the part you'll replace with a from-scratch sparse-optical-flow implementation.
- **Why include this**: BoT-SORT was state-of-the-art on MOTChallenge at publication and is widely cited as the production benchmark in tracking-by-detection.

### 7.5 Optional v1.1: Deep OC-SORT

- **Paper**: Maggiolino, Ahmad, Cao, Kitani. "Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification." ICIP 2023. arXiv:2302.11813.
- **Status**: Not in v1.0. Deep OC-SORT requires appearance features from a CV embedding model. The library will ship the `Embedder` interface and document how to wire any embedding source to Deep OC-SORT's logic, but a complete implementation depends on shipping or referencing a model.

### 7.6 Deliberately excluded

| Algorithm | Why excluded |
|---|---|
| MOTIP, SambaMOTR, CO-MOT | End-to-end transformer trackers; different architecture and runtime needs. |
| FairMOT, JDE | Joint detection-and-tracking; out of scope (no detection in this library). |
| StrongSORT | Largely superseded by BoT-SORT in benchmarks; would add maintenance burden for diminishing returns. |
| Centroid trackers | Insufficient accuracy for the "modern family" framing. |

---

## 8. Plugin interfaces — extension points for learned and classical components

The library exposes four plugin interfaces that let users substitute classical components with learned alternatives. Each tracker accepts these as constructor options, with classical defaults provided. This is what positions the library to absorb learning-based motion models, learned association costs, and appearance embeddings as the research matures, without requiring API changes.

### 8.1 Design philosophy

The architecture treats classical components (Kalman filter, IoU cost, Hungarian assignment) and learned components (LSTM motion predictors, learned cost functions, neural Re-ID) as plug-compatible implementations of the same interface. This matters for three reasons.

**First, the research is moving here.** Multiple 2024-2025 papers (MambaTrack, MotionTrack, ETTrack) demonstrate that learned motion predictors outperform Kalman filters on non-linear motion benchmarks like DanceTrack. Specific reported improvements: ETTrack outperforms Kalman by up to 6.5% IDF1 on non-linear motion; MambaTrack matches or exceeds Kalman-based methods on MOT17 with lower computational cost. The interface anticipates this without committing to specific implementations.

**Second, the cost of being wrong is low.** If learned components turn out to be slower in practice on browser/edge hardware (which is the current reality), nothing about the library changes — the classical defaults are always available. If learned components win, contributors can ship adapter packages without modifying the core library.

**Third, it positions the library architecturally.** "A tracker library with pluggable motion models, cost functions, association strategies, and embedders" is a stronger pitch than "a tracker library with hardcoded Kalman filtering." The interfaces are the moat.

### 8.2 The four plugin interfaces

#### 8.2.1 `MotionPredictor`

Replaces or augments the Kalman filter. Default: `KalmanCvBBox` (constant-velocity bounding-box model from SORT) or `KalmanCvXyah` (constant-velocity center-aspect-height model from DeepSORT).

```ts
interface MotionPredictor<TState = unknown> {
  /** Initialize state from a first observation. */
  init(detection: Detection): TState;

  /** Predict state forward one frame, no measurement. */
  predict(state: TState): TState;

  /**
   * Incorporate a new measurement into state.
   * For Kalman: this is the update step.
   * For learned models: this is the conditioning step on the latest observation.
   */
  update(state: TState, detection: Detection): TState;

  /** Extract the predicted bbox from state, for cost matrix construction. */
  toBBox(state: TState): BBox;

  /**
   * Optional: extract a velocity or motion direction vector from state.
   * Used by OC-SORT's observation-centric momentum (OCM).
   * Implementations that don't model velocity should return null.
   */
  toVelocity?(state: TState): [number, number] | null;
}
```

The generic `TState` parameter lets each implementation choose its own state representation. A Kalman filter uses a state vector + covariance matrix. An LSTM-based predictor uses a hidden state tensor. The tracker never inspects the state directly.

Worked example — a hypothetical learned motion predictor wrapping Transformers.js:

```ts
class LearnedMotionPredictor implements MotionPredictor<LstmState> {
  constructor(private model: TransformersJsModel) {}

  init(det) { return { hidden: zeros(128), lastObs: det.bbox }; }
  predict(state) { return this.model.step(state); }  // async would require AsyncTracker
  update(state, det) { return { ...state, lastObs: det.bbox }; }
  toBBox(state) { return state.lastObs; }
}
```

The library does not ship this. It documents the interface and ships only the classical defaults.

#### 8.2.2 `CostFunction`

Replaces or augments the IoU-based cost matrix between predicted tracks and detections. Default: `IoUCost`. The library also ships `GIoUCost` for OC-SORT compatibility.

```ts
interface CostFunction {
  /**
   * Compute an M×N cost matrix between M predicted tracks and N detections.
   * Returns a Float64Array of length M*N in row-major order.
   * Values must be in [0, 1] for IoU-like costs (1 = no overlap).
   * Use Number.POSITIVE_INFINITY for gated/forbidden pairs.
   */
  compute(
    predictions: ReadonlyArray<BBox>,
    detections: ReadonlyArray<Detection>,
    context?: AssociationContext
  ): Float64Array;
}

interface AssociationContext {
  /** Track velocities, if available, in the same order as predictions. */
  velocities?: ReadonlyArray<[number, number] | null>;
  /** Optional Re-ID embeddings for predictions and detections. */
  trackEmbeddings?: ReadonlyArray<Float32Array | null>;
  detEmbeddings?: ReadonlyArray<Float32Array | null>;
  /** Track ages, for distance gating in long-occluded tracks. */
  trackAges?: ReadonlyArray<number>;
}
```

The `AssociationContext` is the key extension. A custom cost function can fuse IoU with appearance similarity, motion compatibility, or any other signal — without the tracker needing to know what's being combined.

Worked example — fused IoU + appearance cost as used in BoT-SORT-ReID:

```ts
class FusedIoUReIdCost implements CostFunction {
  constructor(private iouWeight = 0.5, private reidWeight = 0.5) {}

  compute(preds, dets, ctx) {
    const iou = computeIoUMatrix(preds, dets);
    if (!ctx?.trackEmbeddings || !ctx?.detEmbeddings) return iou;
    const reid = computeCosineDistanceMatrix(ctx.trackEmbeddings, ctx.detEmbeddings);
    return blend(iou, reid, this.iouWeight, this.reidWeight);
  }
}
```

#### 8.2.3 `AssociationStrategy`

Replaces or augments Hungarian assignment. Default: `HungarianAssociator` using the in-tree Jonker-Volgenant implementation. Most users will never need to swap this, but it's exposed for completeness and to allow experimental strategies (e.g., greedy assignment for very large scenes where O(n³) is prohibitive).

```ts
interface AssociationStrategy {
  /**
   * Given an M×N cost matrix (row-major Float64Array), produce matched pairs
   * and the unmatched indices.
   */
  associate(
    cost: Float64Array,
    M: number,
    N: number,
    options?: { costThreshold?: number }
  ): AssociationResult;
}

interface AssociationResult {
  /** Matched pairs as [trackIndex, detectionIndex] tuples. */
  matched: Array<[number, number]>;
  unmatchedTracks: number[];
  unmatchedDetections: number[];
}
```

The library ships `HungarianAssociator` (optimal) and `GreedyAssociator` (faster but suboptimal, useful for very large M×N). Custom strategies are rare but possible.

#### 8.2.4 `Embedder`

Provides appearance embeddings for Re-ID. Used by `BotSortTracker` with `withReid: true`, and by any custom `CostFunction` that needs appearance similarity. The library does not ship a default — Re-ID requires a model, and shipping a model violates the zero-dependency principle.

```ts
interface Embedder {
  /** Embedding dimensionality (e.g. 128, 256, 512). */
  readonly dim: number;

  /**
   * Given an image and bounding boxes, return one L2-normalized embedding per box.
   * The image type is `unknown` to keep the interface environment-agnostic
   * (browser ImageBitmap, Canvas, Node Buffer, etc.). Implementers handle this.
   * Async because real Re-ID models run in worker threads or inference runtimes.
   */
  embed(image: unknown, bboxes: ReadonlyArray<BBox>): Promise<Float32Array[]>;
}
```

The library expects users to bring their own Re-ID model via Transformers.js, ONNX Runtime Web, MediaPipe, or a remote API. A separate community package can ship an `Embedder` adapter for common models (OSNet, FastReID variants) without polluting the core library.

#### 8.2.5 `CmcProvider`

Estimates camera motion between consecutive frames. Used by `BotSortTracker` to correct track predictions for camera movement. Default: `SparseOpticalFlowCmc` (Shi-Tomasi corner detection + Lucas-Kanade tracking, both implemented in pure TS from scratch).

```ts
interface CmcProvider {
  /**
   * Estimate the affine warp from previous frame to current frame.
   * Returns a 2x3 affine matrix as a 6-element Float64Array (row-major),
   * or null if estimation failed (e.g., insufficient features).
   */
  estimate(prevFrame: unknown, currFrame: unknown): Promise<Float64Array | null>;
}
```

Users who want ECC (the technique used in the original BoT-SORT paper, via OpenCV's `findTransformECC`) can implement it themselves and pass it as a plugin.

### 8.3 Synchronous vs. asynchronous trackers

Classical components are synchronous and take microseconds per call. Learned components are asynchronous (you `await` an inference call) and take milliseconds. This is a fundamental tension that affects the public API.

The library resolves this with two parallel tracker families:

- **Synchronous trackers** (`SortTracker`, `ByteTracker`, `OcSortTracker`, `BotSortTracker`): accept synchronous-only plugins. Fast, deterministic, suitable for high-FPS scenarios. `update()` returns `Track[]` directly.
- **Asynchronous trackers** (`AsyncByteTracker`, `AsyncOcSortTracker`, `AsyncBotSortTracker`): accept async plugins like learned motion predictors or Re-ID embedders. `update()` returns `Promise<Track[]>`.

The constructor signatures differ slightly (async trackers accept the async plugin interfaces), but the algorithmic logic is shared via the base classes. Users who only use classical components pay no async cost; users who plug in learned components opt into the async path explicitly.

This is shipped in v0.3 or later. v0.1 ships synchronous trackers only.

### 8.4 What's deliberately not pluggable

- **Track lifecycle states** (`tentative`, `confirmed`, `lost`, `removed`). These are part of the core data model, not a strategy. If you want different lifecycle semantics, you write a new tracker.
- **Detection input format.** Bounding boxes in `xyxy` are canonical. The library does not provide a `DetectionAdapter` interface for non-bbox detections (e.g., point trackers, polygon trackers). That's a different library.
- **The frame loop.** The library is pull-based by design. No plugin can change this.

### 8.5 Why plugins do not invert dependencies

The user instantiates the plugin and passes it to the tracker constructor. The tracker holds a reference. The library does not provide a plugin registry, dependency injection container, or service locator. This kind of abstraction is unwarranted for a library this size and would obscure what's actually happening at construction time.

---

## 9. Performance strategy

### 9.1 Targets

The honest target is **single-thread JS performance competitive with native Python+NumPy on realistic scene complexity**, not against C++ or CUDA implementations.

Specific numeric goals for v1.0, measured on a 2024-vintage laptop:

| Scenario | Target |
|---|---|
| 10 tracks × 10 detections per frame, ByteTracker | ≥ 5,000 FPS |
| 50 tracks × 50 detections per frame, ByteTracker | ≥ 500 FPS |
| 100 tracks × 100 detections per frame, BotSortTracker (no ReID) | ≥ 100 FPS |

These targets are aggressive but achievable. Reference: the OC-SORT paper reports 700+ FPS on CPU for moderate scenes; matching half of that in JS is a stretch goal.

### 9.2 Memory layout

- **Bounding boxes as typed arrays**, not object arrays. A scene with 100 detections fits in a 1.6 KB `Float64Array` rather than 100 heap-allocated objects.
- **Cost matrices as flat `Float64Array`**, row-major. Hungarian assignment operates directly on this representation.
- **Track state stored as struct-of-arrays internally** for trackers with many tracks. The public-facing `Track` object is materialized lazily at `update()` return time.

### 9.3 Allocation discipline

Per-frame allocation is the dominant performance cost in V8 for tight loops. The strategy:

- **Pre-allocated scratch buffers** sized to the maximum observed scene complexity, grown lazily but never shrunk.
- **Object pooling** for `Track` instances returned to users (with documented immutability contract on the return value).
- **No closures inside `update()`**. Inner-loop callbacks are hoisted to module-scope functions.

### 9.4 Algorithmic hot paths

The three hot paths, in expected cost order:

1. **Cost matrix construction** — O(MN) per frame. Optimized with batched IoU on flat arrays.
2. **Hungarian assignment** — O(min(M, N)³) worst case. The Jonker-Volgenant choice matters here.
3. **Kalman prediction** — O(state_dim²) per track. State_dim is 7 or 8, so the constant matters more than asymptotics; unrolled loops win.

### 9.5 What we deliberately won't do for v1.0

- **WebAssembly hot paths.** Profile first. If pure-JS Hungarian assignment is the bottleneck at scale, a WASM implementation is a v1.1 optimization. Adding a WASM build step before measuring is premature optimization that complicates the contributor story.
- **SIMD via WebAssembly SIMD or `Float64x2`.** Same reasoning. Measure first.
- **Web Workers in core.** The library is single-threaded by design. Users who want worker offload wrap the tracker themselves.
- **GPU acceleration.** Out of scope for tracking-by-detection trackers; the algorithms are not embarrassingly parallel.

---

## 10. Accuracy strategy and benchmarking

### 10.1 The reality of cross-implementation accuracy

Bit-for-bit equivalence with Python references is not achievable and not the goal. Floating-point order of operations, Hungarian tie-breaking, and Kalman initialization choices all introduce small divergences that accumulate into different track ID assignments on long sequences. The goal is **HOTA/MOTA/IDF1 numbers within a small, documented delta** of the published reference numbers.

### 10.2 Reference selection

For each algorithm, the canonical Python reference is fixed:

| Algorithm | Reference repository | Reference commit |
|---|---|---|
| SORT | abewley/sort | latest stable tag |
| ByteTrack | FoundationVision/ByteTrack | latest stable tag |
| OC-SORT | noahcao/OC_SORT | latest stable tag |
| BoT-SORT | NirAharon/BoT-SORT | latest stable tag |

The commit hash is recorded in the benchmark report. Reference implementations are not vendored; they are referenced via documentation.

### 10.3 Benchmark suite

The eval package implements:

- **MOT17/MOT20 sequence loaders.** Detections are loaded from publicly-available pre-computed detection files (the same ones the reference papers use, e.g. the YOLOX-X detections shipped with ByteTrack). Using the same detections is essential; otherwise tracker performance differences are confounded with detector performance differences.
- **HOTA, MOTA, IDF1, MT, ML, IDsw, Frag metrics.** Implemented from the IJCV 2020 HOTA paper (Luiten et al.) and the CLEAR-MOT and ID metrics papers. HOTA in particular is non-trivial — the official `TrackEval` repository (JonathonLuiten/TrackEval) is the reference. The eval package is allowed Node-only dev dependencies for things like CSV parsing, but the metric computations themselves are zero-dependency.
- **JSON-formatted result files** matching the MOTChallenge submission format, for direct comparison with the public MOTChallenge leaderboard.

### 10.4 Acceptance criteria for an algorithm to be considered "implemented"

| Metric | Acceptance window vs. published number |
|---|---|
| HOTA | within ±1.0 absolute |
| MOTA | within ±1.5 absolute |
| IDF1 | within ±1.5 absolute |

If a tracker fails the window, it's not released. Differences within the window are documented in the benchmark report with attribution (e.g. "JV vs. Munkres tie-breaking accounts for 0.3 HOTA").

### 10.5 Regression testing

A fixed test corpus of detection sequences (a subset of MOT17-04, MOT17-09, MOT17-11) has its tracker outputs snapshotted in the repository. CI verifies that the output is byte-identical to the snapshot. Any change that alters the snapshot must be intentional, justified in the PR, and re-baselined with new HOTA numbers.

### 10.6 Cross-implementation reporting

The benchmark report includes a transparent comparison table:

```
              HOTA    MOTA    IDF1    Δ HOTA vs. published
SORT      :   45.8    65.4    62.1    +0.2
ByteTrack :   62.7    79.9    76.8    -0.4
OC-SORT   :   62.9    78.5    77.1    -0.3
BoT-SORT  :   64.6    80.1    79.7    -0.4
```

Honest about deltas, transparent about causes. This is the kind of disclosure that signals "real implementation" to anyone evaluating the project.

---

## 11. Public API surface

The package is published to npm as `vestige.js`. All examples below use that as the import source.

### 11.1 Tracker construction — defaults only

```ts
import { ByteTracker } from 'vestige.js';

const tracker = new ByteTracker({
  trackThresh: 0.5,
  matchThresh: 0.8,
  trackBuffer: 30,
  frameRate: 30,
});
```

Every option is explicit. There is no `pipeline()` factory, no `createTracker()` magic, no string-to-class dispatch. The user knows exactly which class they're instantiating. The classical plugins (`KalmanCvXyah` motion predictor, `IoUCost` cost function, `HungarianAssociator`) are used as defaults; the user doesn't see them.

### 11.2 Per-frame usage

```ts
// In the user's frame loop:
const detections = await detector.detect(frame);
const tracks = tracker.update(detections);
// tracks is Track[] — render, log, etc.
```

The output array contains only confirmed tracks. Tentative and lost tracks are inspectable via the dedicated methods.

### 11.3 State inspection

```ts
tracker.getActiveTracks();   // currently matched
tracker.getLostTracks();     // matched recently but not this frame
tracker.frameIndex;          // current frame counter
tracker.reset();             // back to initial state
```

### 11.4 Plugin wiring — classical components

Users who want to swap classical components (e.g. use GIoU cost instead of IoU, or use the SORT-style motion model in ByteTrack) do so explicitly:

```ts
import {
  ByteTracker,
  KalmanCvBBox,
  GIoUCost,
} from 'vestige.js';

const tracker = new ByteTracker({
  trackThresh: 0.5,
  motionPredictor: new KalmanCvBBox(),  // override default
  costFunction: new GIoUCost(),         // override default
});
```

### 11.5 Plugin wiring — Re-ID and CMC (BoT-SORT)

```ts
import { BotSortTracker, SparseOpticalFlowCmc } from 'vestige.js';

const tracker = new BotSortTracker({
  trackThresh: 0.5,
  cmc: new SparseOpticalFlowCmc(),
  embedder: myCustomEmbedder,  // user-provided, wrapping their preferred runtime
  withReid: true,
});
```

### 11.6 Async tracker variants (v0.3+) — learned components

Users plugging in learned motion predictors or async embedders use the async tracker family:

```ts
import { AsyncByteTracker } from 'vestige.js/async';
import { MyLearnedMotionPredictor } from '@my-org/vestige-learned-motion';  // hypothetical community package

const tracker = new AsyncByteTracker({
  trackThresh: 0.5,
  motionPredictor: new MyLearnedMotionPredictor(modelUrl),
});

// Note: update() is async
const tracks = await tracker.update(detections);
```

The async family exists specifically so that users who only need classical components don't pay async overhead.

### 11.7 Exported geometry utilities

The geometry primitives are also exported, because they're useful to users:

```ts
import { iou, giou, bboxArea } from 'vestige.js/geometry';
```

This is a one-time decision: exposing utilities signals "this library is a toolkit, not a black box."

### 11.8 What is deliberately not in the public API

- No singleton tracker instance.
- No global configuration object.
- No automatic Web Worker support (users wrap manually).
- No `.toJSON()` / `.fromJSON()` serialization in v1.0 (deferred to v1.2).
- No mixed sync/async API. The tracker variant chosen at construction time determines whether `update()` is sync or async; the library does not auto-detect.

---

## 12. Build, packaging, and distribution

### 12.1 Build tooling

- **Build tool**: `tsup` or `tshy`. Both produce dual ESM/CJS output with TypeScript declarations from a single config.
- **TypeScript target**: ES2020. Wide enough support (Node 14+, all evergreen browsers).
- **Module format**: ESM-first. CJS output for Node compatibility, marked as legacy.
- **Bundle target**: Pure JS, no external runtime. Tree-shakable — importing `SortTracker` should not pull in `BotSortTracker` code.

### 12.2 Package metadata

```jsonc
{
  "name": "vestige.js",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "require": "./dist/index.cjs",
      "types": "./dist/index.d.ts"
    },
    "./geometry": "./dist/geometry/index.js"
  },
  "sideEffects": false,
  "dependencies": {},
  "engines": { "node": ">=14" }
}
```

**Note the empty `dependencies` block.** This is the constitutional commitment.

### 12.3 Bundle size budget

| Package | Minified | Gzipped |
|---|---|---|
| `vestige.js` (full) | ≤ 80 KB | ≤ 25 KB |
| Single tracker (e.g. just ByteTracker, tree-shaken) | ≤ 35 KB | ≤ 12 KB |

These budgets are enforced in CI via `size-limit` (a dev dependency).

### 12.4 Versioning

Strict semver. The major version is reserved for breaking API changes — and the design principle of "algorithms own their options" means adding new trackers is non-breaking, which matters for contributor experience.

---

## 13. Testing strategy

### 13.1 Test layers

1. **Unit tests** — pure functions in geometry, solvers, filters. Hand-built input/output cases verified against scipy/numpy oracle values computed offline.
2. **Property-based tests** — `fast-check` (dev dependency only). Examples: Hungarian assignment cost is never higher than any specific assignment; track IDs are stable when detections are stable; lost tracks are eventually reaped.
3. **Integration tests** — full tracker runs on synthetic sequences (e.g. linear motion, occlusion, ID swap scenarios) with hand-verified expected outputs.
4. **Snapshot regression tests** — fixed input sequences with snapshotted outputs. Detects accidental drift.
5. **MOTChallenge benchmark tests** — run as part of release CI, comparing against published numbers.

### 13.2 Test infrastructure

- **Test runner**: Vitest. Fast, ESM-native, good TypeScript support, mature.
- **Property testing**: fast-check.
- **Snapshot library**: Vitest's built-in inline snapshots for small cases; file snapshots for large sequences.
- **Benchmark runner**: `mitata` or `tinybench` — both have minimal footprints and are dev-only.

### 13.3 Cross-implementation validation

A dedicated `validation/` directory contains:

- Pre-computed Python reference outputs on fixed input sequences (committed as test fixtures).
- Tests that load the same input through the TS tracker and compare outputs.
- Documented expected divergences (e.g. "frames 47-62 show different track IDs due to JV vs. Munkres tie-breaking on a 4-way tie").

This is the artifact that turns "I implemented these papers" into "I implemented them correctly, and here's evidence."

---

## 14. Contributor model

### 14.1 Why this project welcomes contributors

The architecture exposes natural extension points where contributors can do bounded, high-leverage work:

| Extension point | Example contributor PR |
|---|---|
| New tracker variant | "Add StrongSORT" / "Add Hybrid-SORT" |
| New cost function | "Add EIoU / Focal-EIoU cost" |
| New motion model | "Add constant-acceleration model for high-speed scenarios" |
| New CMC provider | "Add ECC-based CMC implementation" |
| New Re-ID adapter | "Add OSNet wrapper for Transformers.js" |
| Performance optimization | "Replace inner loop with SIMD on Hungarian assignment" |
| Benchmark on new dataset | "Add DanceTrack benchmark numbers" |

Each of these is a well-scoped issue that doesn't require coordinating with the maintainer on API design. That's deliberately by design.

### 14.2 Governance

For v0.x and v1.0: single-maintainer with strong documentation. For v2+: an open RFC process for new tracker additions, with a published acceptance criteria checklist (paper exists, HOTA within window, doesn't add runtime dependencies).

### 14.3 Documentation strategy

- **Algorithm docs** that explain *why*, not just *how*. Each tracker's docs include the paper summary, the failure mode it addresses, and a "when to use this" recommendation.
- **Architecture docs** (this document) kept current as the source of design truth.
- **Decision log** in `docs/decisions/` recording why each non-obvious choice was made (JV vs. Munkres, Cholesky vs. naive inverse, etc.). Anyone joining the project gets context, not opinions.

---

## 15. Roadmap and milestones

### 15.1 v0.1 — SORT + ByteTrack (months 1–3)

**Goal**: prove the core architecture and ship the two simplest trackers correctly.

- Numerical core: bbox geometry, IoU/GIoU, Jonker-Volgenant, Kalman filter with cv-bbox model.
- BaseTracker lifecycle.
- SortTracker and ByteTracker.
- Test suite: unit, property, snapshot.
- MOT17 benchmark for both trackers — HOTA within published window.
- Documentation site (Vitepress) with first-pass content.
- Browser example: webcam + YOLOv8 via Transformers.js + ByteTracker.
- Published to npm as `vestige.js@0.1.0`.

### 15.2 v0.2 — OC-SORT (months 4–6)

**Goal**: ship the first non-trivial tracker, demonstrating depth.

- ORU, OCM, OCR implementations.
- OcSortTracker with full hyperparameter surface.
- Extended Kalman filter motion model variants.
- MOT17 + MOT20 + DanceTrack benchmark numbers.
- Snapshot tests against the official Python reference output.
- Decision log entries for the non-obvious ORU math.

### 15.3 v0.3 — BoT-SORT + full plugin architecture (months 7–9)

**Goal**: state-of-the-art tracker, complete extension architecture, async tracker variants for learned components.

- BotSortTracker.
- Full plugin interface surface: `MotionPredictor`, `CostFunction`, `AssociationStrategy`, `Embedder`, `CmcProvider`.
- All classical default implementations shipped (`KalmanCvBBox`, `KalmanCvXyah`, `IoUCost`, `GIoUCost`, `HungarianAssociator`, `GreedyAssociator`, `SparseOpticalFlowCmc`).
- Async tracker variants (`AsyncByteTracker`, `AsyncOcSortTracker`, `AsyncBotSortTracker`) for learned-plugin support.
- Worked example: BoT-SORT with a learned motion predictor wrapping a Transformers.js model, replacing the Kalman filter. Published as a blog post and an example in `examples/learned-motion-demo/`.
- Full benchmark suite published.
- Documentation site v2 with comparison tables, decision flowchart, and a "writing your own plugin" guide.

### 15.4 v1.0 — Production release (months 10–12)

**Goal**: API stability commitment.

- API frozen.
- Bundle size budget enforced.
- Cross-implementation validation suite complete.
- Performance baselines published.
- Migration guide from any prior 0.x.

### 15.5 Post-v1.0 candidates

In rough priority order:

- Deep OC-SORT support (requires `Embedder` ecosystem maturity).
- Community-contributed learned motion predictors as separate `@org/lib-*` packages (MambaTrack-style, ETTrack-style).
- WebAssembly hot paths if profiling justifies.
- JSON serialization for state save/load.
- Track interpolation utilities (off by default; some users want gap-filling).
- 3D tracking via ByteTrackV2-style extension.

---

## 16. Risks and open questions

### 16.1 The reference Python implementations diverge from each other

When the ultralytics implementation of ByteTrack reports different numbers than the official FoundationVision one, which is "right"? The chosen mitigation: use the official paper authors' implementations as canonical, document version pins, and report against published paper numbers as the primary anchor.

### 16.2 Detection set must match for benchmarks to be meaningful

Tracker performance is heavily dominated by detection quality. To fairly compare against published numbers, the same detection sets (pre-computed YOLOX-X detections used in the ByteTrack paper) must be used. These are public but large — the eval package downloads them on demand rather than vendoring.

### 16.3 Browser bench environments are noisy

Performance numbers measured in the browser vary by 20-30% across V8 versions, GC pressure, and tab visibility state. Benchmark methodology must include warmup runs, GC pinning where possible, and median-of-N reporting. Node bench numbers are more stable and will be the primary published numbers, with browser numbers reported separately as "representative."

### 16.4 The market may move toward end-to-end trackers

YOLO26 with native NMS-free inference (Jan 2026) and transformer-based MOT trackers (MOTIP, SambaMOTR, CO-MOT) hint at a future where tracking-by-detection is the legacy approach. The library accepts this risk — tracking-by-detection remains the production standard in AV stacks and will for years, and the codebase is small enough that maintenance cost is bounded.

### 16.5 What "modern family" means may shift

If a new state-of-the-art tracker is published that decisively beats BoT-SORT on MOT and DanceTrack, the library's "modern family" claim weakens. The contributor model is the hedge: someone can submit a PR for the new tracker, and the architecture supports adding it without API churn.

### 16.6 Single-maintainer bus factor

For v0.x and v1.0, the project lives or dies on a single maintainer's attention. The mitigation is documentation discipline — every non-obvious choice is recorded so a future contributor can resume the work. The decision log is the most important artifact for this.

---

## 17. Reference appendix

### 17.1 Primary papers

1. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). **Simple Online and Realtime Tracking.** ICIP. arXiv:1602.00763.
2. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). **ByteTrack: Multi-Object Tracking by Associating Every Detection Box.** ECCV. arXiv:2110.06864.
3. Cao, J., Pang, J., Weng, X., Khirodkar, R., & Kitani, K. (2023). **Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking.** CVPR. arXiv:2203.14360.
4. Aharon, N., Orfaig, R., & Bobrovsky, B. (2022). **BoT-SORT: Robust Associations Multi-Pedestrian Tracking.** arXiv:2206.14651.
5. Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2020). **HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking.** IJCV. arXiv:2009.07736.

### 17.2 Supporting papers

6. Wojke, N., Bewley, A., & Paulus, D. (2017). **Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT).** ICIP. arXiv:1703.07402. *(Source of the cv-xyah motion model.)*
7. Maggiolino, G., Ahmad, A., Cao, J., & Kitani, K. (2023). **Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification.** ICIP. arXiv:2302.11813. *(Reference for the post-v1.0 Deep OC-SORT support.)*
8. Zhang, Y., Wang, X., Ye, X., Zhang, W., Lu, J., Tan, X., Ding, E., Sun, P., & Wang, J. (2023). **ByteTrackV2: 2D and 3D Multi-Object Tracking by Associating Every Detection Box.** arXiv:2303.15334. *(Clearest writeup of ByteTrack association logic.)*
9. Dendorfer, P., Osep, A., Milan, A., Schindler, K., Cremers, D., Reid, I., Roth, S., & Leal-Taixé, L. (2020). **MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking.** *(Benchmark methodology and dataset reference.)*
10. Bernardin, K., & Stiefelhagen, R. (2008). **Evaluating multiple object tracking performance: the CLEAR MOT metrics.** EURASIP. *(Reference for MOTA and MOTP.)*
11. Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016). **Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking.** ECCV Workshop. *(Reference for IDF1.)*

### 17.3 Algorithmic primitives

12. Jonker, R., & Volgenant, A. (1987). **A shortest augmenting path algorithm for dense and sparse linear assignment problems.** *Computing*, 38(4), 325-340. *(The assignment algorithm.)*
13. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). **Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression.** CVPR. *(GIoU.)*
14. Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). **Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.** AAAI. *(DIoU/CIoU.)*

### 17.4 Official reference implementations

- abewley/sort — original SORT (Python).
- nwojke/deep_sort — DeepSORT (Python). *(Reference for cv-xyah motion model.)*
- FoundationVision/ByteTrack — official ByteTrack (Python, MIT).
- noahcao/OC_SORT — official OC-SORT (Python, MIT).
- NirAharon/BoT-SORT — official BoT-SORT (Python, MIT).
- GerardMaggiolino/Deep-OC-SORT — official Deep OC-SORT (Python).
- JonathonLuiten/TrackEval — canonical HOTA implementation.

### 17.5 Benchmarks and datasets

- **MOT17** — https://motchallenge.net/data/MOT17/
- **MOT20** — https://motchallenge.net/data/MOT20/
- **DanceTrack** — https://github.com/DanceTrack/DanceTrack
- **TrackEval** — https://github.com/JonathonLuiten/TrackEval

---

## 18. Naming

### 18.1 The name: vestige.js

**Package name (npm):** `vestige.js`
**Repository name (GitHub):** `vestige.js`
**Working title in documentation:** vestige.js

### 18.2 Why this name

A vestige is a persistent trace of something across time — a small remaining portion of what was once whole, evidence that something existed and continues to exist in some attenuated form. That is structurally what a track is in multi-object tracking: an identity preserved through occlusion, frame drops, motion, and appearance changes. The metaphor maps onto the function rather than describing it generically, which is the same naming pattern Tryolabs used when they named their Python MOT library `norfair`.

The `.js` suffix is a deliberate signal that this is a JavaScript/TypeScript library, consistent with established conventions in the JS computer vision and ML space: `tracking.js`, `face-api.js`, `Transformers.js`, `tensorflow.js`. The suffix costs three characters but earns immediate recognition of the runtime to anyone scanning a list of dependencies, search results, or blog post mentions.

### 18.3 Discoverability assessment

The name does not lead with "track" or "mot" — the stems that would have given it instant in-domain recognition. This is a deliberate trade-off. Both stems are heavily squatted on npm and GitHub, and the bare-stem candidates either had real collisions (e.g., the `byte-track` org, `coding-blocks/motley`) or near-misses that would muddy SEO results.

The discoverability work that the name does *not* do has to be carried by README copy, npm `keywords`, GitHub topics, and the README's opening sentence:

```jsonc
// package.json
{
  "name": "vestige.js",
  "keywords": [
    "object-tracking", "multi-object-tracking", "mot", "tracking",
    "bytetrack", "sort", "ocsort", "botsort",
    "computer-vision", "typescript", "browser"
  ]
}
```

GitHub topics on the repository should mirror this list.

### 18.4 Verification at time of selection

The name was verified clear on both registries at the time of this decision:

- **npm**: `https://registry.npmjs.org/vestige.js` returned 404 (available).
- **GitHub repository search**: `q=vestige.js+in:name` returned zero exact matches and zero near-misses.
- **Bare `vestige` on GitHub**: taken by `samvallad33/vestige` (an FSRS-6 spaced-repetition library for AI agents), but the `.js`-suffixed namespace is unaffected.

### 18.5 Pre-publish checklist

Before pushing the initial repository or `npm publish`, re-verify and reserve:

1. `npm view vestige.js` — confirms still 404 (registry checks have minute-level staleness).
2. Browse to `https://github.com/<yourhandle>/vestige.js` — confirms the repo name is not taken in your account.
3. Search Google for `"vestige.js"` — checks for stale SEO conflicts.
4. Reserve the npm name immediately with a placeholder `package.json` and stub README, so it doesn't get squatted during development.
5. Push the empty repo to GitHub with a one-paragraph README so the GitHub repo URL is reserved alongside.

---

## Document metadata

This is a living document. Material changes should be accompanied by a changelog entry and, where they affect public API or algorithmic behavior, a decision log entry in `docs/decisions/`.
