import type { KalmanState } from '../filters/kalman.js';
import type { BBox, Detection, Track, Tracker, TrackState } from '../types.js';

/**
 * Per-frame book-keeping for a single track inside a {@link BaseTracker}.
 *
 * The library never exposes this shape directly — `update()` materializes a
 * public {@link Track} from it. The fields are mutated in place by the
 * per-track primitives ({@link BaseTracker.applyMatch},
 * {@link BaseTracker.applyMiss}) and by each tracker's lifecycle transitions;
 * this is the one place the codebase deliberately uses mutation, because the
 * tracker's per-frame state is what makes it not a pure function.
 *
 * Subclasses are responsible for keeping {@link InternalTrack.kalmanState} and
 * {@link InternalTrack.bbox} consistent: `bbox` is the bbox derived from the
 * current `kalmanState` (after the latest predict or update). Caching `bbox`
 * here means `materializeTrack` doesn't need to know the motion-model specifics.
 */
export interface InternalTrack<TPayload = unknown> {
  /** Stable identifier assigned at creation; never changes. */
  id: number;
  /** Lifecycle phase; see {@link TrackState}. */
  state: TrackState;
  /** Total frames since the track was created (including unmatched frames). */
  age: number;
  /** Total frames the track has been matched (never decreases). */
  hits: number;
  /**
   * Consecutive matched frames since the last miss; reset to 0 on miss,
   * incremented on match. Used by trackers (e.g. SORT) whose confirmation
   * criterion is "N matches in a row" rather than "N matches total."
   */
  hitStreak: number;
  /** Consecutive frames since last successful match; 0 means matched this frame. */
  timeSinceUpdate: number;
  /** Kalman mean + covariance, post-latest-predict-or-update. */
  kalmanState: KalmanState;
  /** Bbox derived from `kalmanState`; refreshed by predictTrack and updateTrack. */
  bbox: BBox;
  /**
   * Most-recent detection the track was matched to. Used at export time to
   * carry score / classId / payload through to the public {@link Track}.
   */
  lastDetection: Detection<TPayload>;
}

/**
 * Output of an association pass. Indices refer to the parallel `detections`
 * and `associableTracks` arrays passed into association — not to track IDs or
 * to the full `tracks` map.
 *
 * Every detection index appears exactly once across `matched[*][1]` and
 * `unmatchedDetections`; every track index appears exactly once across
 * `matched[*][0]` and `unmatchedTracks`. Callers rely on this invariant when
 * materializing the next frame.
 */
export interface AssociationResult {
  /** Matched pairs `[trackIndex, detectionIndex]`. */
  readonly matched: ReadonlyArray<readonly [number, number]>;
  /** Track indices (into the associable-tracks array) that did not match. */
  readonly unmatchedTracks: ReadonlyArray<number>;
  /** Detection indices (into `detections`) that did not match a track. */
  readonly unmatchedDetections: ReadonlyArray<number>;
}

/**
 * Options consumed by {@link BaseTracker.runStandardLifecycle}. Each tracker
 * that opts into the standard lifecycle (currently only `SortTracker`) supplies
 * these from its own resolved option fields; `BaseTracker` itself stores no
 * options, because the four trackers in the family don't share a meaningful
 * option surface (see ARCHITECTURE.md §2.2).
 */
export interface StandardLifecycleOptions {
  /** Consecutive matches required to promote a tentative track to confirmed. */
  readonly minHits: number;
  /** Frames a `lost` (or `tentative`) track survives without a match before removal. */
  readonly maxAge: number;
}

/**
 * Signature of an association function that drives one association pass —
 * passed as a callback into {@link BaseTracker.runStandardLifecycle} so the
 * lifecycle helper doesn't depend on subclass-specific cost functions. Trackers
 * with multi-stage matching (ByteTrack, OC-SORT, BoT-SORT) skip the helper
 * entirely and orchestrate their own stages.
 */
export type AssociateFn<TPayload> = (
  detections: ReadonlyArray<Detection<TPayload>>,
  associableTracks: ReadonlyArray<InternalTrack<TPayload>>,
) => AssociationResult;

/**
 * Shared infrastructure for every tracker in the SORT / ByteTrack / OC-SORT /
 * BoT-SORT family. See ARCHITECTURE.md §6.1.
 *
 * **What `BaseTracker` owns:**
 *
 * - The `tracks` map, the monotonic `nextId` counter, and the per-frame `_frameIndex`.
 * - The {@link TrackState} enum and the {@link InternalTrack} shape.
 * - Per-track bookkeeping primitives: {@link applyMatch}, {@link applyMiss},
 *   {@link spawnTrack}, {@link sweepRemoved}. These are stateless transforms
 *   over a single track's counters — every tracker in the family wants them.
 * - The {@link runStandardLifecycle} helper that encodes the SORT-default
 *   transitions for trackers that want them.
 * - The {@link exportConfirmed} default rule and {@link materializeTrack}.
 *
 * **What `BaseTracker` does NOT own:**
 *
 * - The `update()` entry point — abstract here. Each tracker's `update()` IS
 *   its algorithm; pretending otherwise was an OOP-style abstraction that
 *   didn't survive contact with ByteTrack's three-stage pipeline.
 * - An abstract `associate()` method — ByteTrack / OC-SORT / BoT-SORT all run
 *   multiple association passes per frame, so a single AssociationResult-shaped
 *   contract isn't useful. Trackers that want SORT-style single-stage
 *   association pass it as a callback into {@link runStandardLifecycle}.
 * - Option storage — each tracker stores its own option fields; the family has
 *   no meaningful shared option surface (ARCHITECTURE.md §2.2).
 *
 * @typeParam TPayload user-defined per-detection payload, threaded unchanged
 *                     through the tracker.
 */
export abstract class BaseTracker<TPayload = unknown> implements Tracker<TPayload> {
  /** All tracks (tentative + confirmed + lost). `removed` tracks are deleted from this map. */
  protected readonly tracks = new Map<number, InternalTrack<TPayload>>();
  /** Monotonic id counter; the next created track gets `nextId++`. Starts at 1. */
  protected nextId = 1;

  protected _frameIndex = 0;

  /** Current frame counter; 0 before any call to `update`, then advances by 1 per call. */
  get frameIndex(): number {
    return this._frameIndex;
  }

  /**
   * Process one frame of detections and return the publicly-visible tracks.
   * Abstract because each tracker's per-frame algorithm IS the tracker —
   * SORT's single-pass association and ByteTrack's three-stage pipeline don't
   * share a template-method shape. See ARCHITECTURE.md §6 for the per-tracker
   * algorithms.
   */
  abstract update(detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[];

  /**
   * Tracks currently in the `confirmed` state, regardless of whether they
   * matched this frame. Use this for inspection / debugging; `update()`'s
   * return value is the right shape for downstream rendering.
   */
  getActiveTracks(): Track<TPayload>[] {
    const out: Track<TPayload>[] = [];
    for (const track of this.tracks.values()) {
      if (track.state === 'confirmed') out.push(this.materializeTrack(track));
    }
    return out;
  }

  /**
   * Tracks currently in the `lost` state (recently confirmed, currently
   * unmatched, still within the per-tracker retention horizon).
   */
  getLostTracks(): Track<TPayload>[] {
    const out: Track<TPayload>[] = [];
    for (const track of this.tracks.values()) {
      if (track.state === 'lost') out.push(this.materializeTrack(track));
    }
    return out;
  }

  /** Clear every track, reset id counter to 1, reset frame counter to 0. */
  reset(): void {
    this.tracks.clear();
    this.nextId = 1;
    this._frameIndex = 0;
  }

  /**
   * Predict one frame forward for an existing track. Subclasses run their
   * motion model's predict step (e.g.
   * {@link import('../filters/kalman.js').KalmanFilter.predict}) and refresh
   * `track.kalmanState` *and* `track.bbox` in place.
   */
  protected abstract predictTrack(track: InternalTrack<TPayload>): void;

  /**
   * Incorporate a new measurement into an existing track. Subclasses run
   * their motion model's update step and refresh both `track.kalmanState`
   * and `track.bbox` in place. `track.lastDetection` is updated here.
   */
  protected abstract updateTrack(
    track: InternalTrack<TPayload>,
    detection: Detection<TPayload>,
  ): void;

  /**
   * Build a fresh {@link InternalTrack} from a first observation. The returned
   * track must have `id = 0` (caller / {@link spawnTrack} assigns the real id),
   * `age = 0`, `hits = 0`, `hitStreak = 0`, `timeSinceUpdate = 0`, and
   * `state = 'tentative'`. `kalmanState` and `bbox` are initialized from
   * `detection`.
   *
   * The `hits = 0` convention matches `abewley/sort` (`sort.py:KalmanBoxTracker.__init__`):
   * the spawn detection is **not** counted as a hit. The first hit is recorded
   * the next time the track matches a detection, via {@link updateTrack} +
   * {@link applyMatch}. Subclasses that diverge from this convention (e.g.
   * counting the init detection as a hit) will confirm tracks one frame
   * earlier than the reference implementation.
   */
  protected abstract initTrack(detection: Detection<TPayload>): InternalTrack<TPayload>;

  /**
   * Apply a successful match: refresh motion state, bump counters, and
   * promote a re-found lost track back to confirmed. Stateless transform over
   * a single (track, detection) pair — every tracker in the family wants this
   * sequence, so it lives here rather than being duplicated in each `update()`.
   *
   * Tracker-specific lifecycle transitions (e.g. tentative→confirmed via
   * `hitStreak ≥ minHits` for SORT, or via stage-3 match for ByteTrack)
   * happen separately in the caller.
   */
  protected applyMatch(track: InternalTrack<TPayload>, detection: Detection<TPayload>): void {
    this.updateTrack(track, detection);
    track.hits++;
    track.hitStreak++;
    track.timeSinceUpdate = 0;
    if (track.state === 'lost') track.state = 'confirmed';
  }

  /**
   * Apply an unsuccessful match: reset the consecutive-hit streak and advance
   * the missed-frame counter. Stateless transform; tracker-specific state
   * transitions (confirmed→lost, lost→removed) happen separately in the caller.
   */
  protected applyMiss(track: InternalTrack<TPayload>): void {
    track.hitStreak = 0;
    track.timeSinceUpdate++;
  }

  /**
   * Build a new track from a detection, assign it the next monotonic id, and
   * register it in the `tracks` map. Returns the registered track so the caller
   * can apply any tracker-specific post-init transitions (e.g. ByteTrack's
   * frame-1 instant activation).
   */
  protected spawnTrack(detection: Detection<TPayload>): InternalTrack<TPayload> {
    const fresh = this.initTrack(detection);
    fresh.id = this.nextId++;
    this.tracks.set(fresh.id, fresh);
    return fresh;
  }

  /** Delete every track currently in the `removed` state from the `tracks` map. */
  protected sweepRemoved(): void {
    for (const [id, track] of this.tracks) {
      if (track.state === 'removed') this.tracks.delete(id);
    }
  }

  /**
   * Run the SORT-default per-frame lifecycle:
   *
   * 1. `_frameIndex++`
   * 2. Predict every existing track; advance `age`.
   * 3. Snapshot non-removed tracks; call `associate` with them and `detections`.
   * 4. {@link applyMatch} each matched pair; {@link applyMiss} each unmatched track;
   *    {@link spawnTrack} each unmatched detection.
   * 5. Advance lifecycle transitions:
   *    - `tentative` → `removed` if `timeSinceUpdate > maxAge`;
   *      → `confirmed` if `hitStreak >= minHits`.
   *    - `confirmed` → `lost` on a miss this frame.
   *    - `lost` → `removed` if `timeSinceUpdate > maxAge`.
   *    The checks chain (not `else if`) so a track can cascade
   *    confirmed → lost → removed in a single pass when `maxAge = 0`,
   *    matching `sort.py`'s `if (trk.time_since_update > self.max_age): pop`.
   * 6. {@link sweepRemoved}.
   * 7. Return {@link exportConfirmed}.
   *
   * This is the lifecycle `abewley/sort` (`sort.py:Sort.update`) uses.
   * ByteTrack / OC-SORT / BoT-SORT have materially different lifecycles
   * (asymmetric `maxAge` between tentative and lost, frame-1 instant
   * activation, observation-centric re-update, etc.) and skip this helper.
   *
   * `associate` is passed as a parameter rather than a method so the helper
   * doesn't depend on a subclass-specific abstract method — that pattern
   * forced ByteTracker to declare a never-called `associate()` stub when
   * this lifecycle was the default `update()` body. Implementations that
   * want zero per-frame allocation should pass a pre-bound arrow-function
   * property (see `SortTracker.associate`).
   */
  protected runStandardLifecycle(
    detections: ReadonlyArray<Detection<TPayload>>,
    options: StandardLifecycleOptions,
    associate: AssociateFn<TPayload>,
  ): Track<TPayload>[] {
    this._frameIndex++;

    // 1. Predict every existing track forward one frame. age++ here so that
    //    tracks created later this frame (in step 4c) correctly start at age 0.
    for (const track of this.tracks.values()) {
      this.predictTrack(track);
      track.age++;
    }

    // 2. Snapshot the associable tracks. Tracks in the `removed` state would
    //    already have been swept at the end of the previous frame, but the
    //    explicit filter guards against subclasses that might inject state.
    const associable: InternalTrack<TPayload>[] = [];
    for (const track of this.tracks.values()) {
      if (track.state !== 'removed') associable.push(track);
    }

    // 3. Associate predictions ↔ detections.
    const { matched, unmatchedTracks, unmatchedDetections } = associate(detections, associable);

    // 4a. Update matched tracks.
    for (const [ti, di] of matched) {
      const track = associable[ti];
      const det = detections[di];
      if (!track || !det) continue;
      this.applyMatch(track, det);
    }

    // 4b. Mark unmatched tracks.
    for (const ti of unmatchedTracks) {
      const track = associable[ti];
      if (!track) continue;
      this.applyMiss(track);
    }

    // 4c. Spawn a new tentative track per unmatched detection.
    for (const di of unmatchedDetections) {
      const det = detections[di];
      if (!det) continue;
      this.spawnTrack(det);
    }

    // 5. Advance lifecycle states. See class JSDoc for the cascade semantics.
    for (const track of this.tracks.values()) {
      if (track.state === 'tentative') {
        if (track.timeSinceUpdate > options.maxAge) {
          track.state = 'removed';
        } else if (track.hitStreak >= options.minHits) {
          track.state = 'confirmed';
        }
      }
      if (track.state === 'confirmed' && track.timeSinceUpdate > 0) {
        track.state = 'lost';
      }
      if (track.state === 'lost' && track.timeSinceUpdate > options.maxAge) {
        track.state = 'removed';
      }
    }

    // 6. Sweep removed tracks out of the map.
    this.sweepRemoved();

    // 7. Export.
    return this.exportConfirmed();
  }

  /**
   * Materialize the public {@link Track[]} for return from `update()`. The
   * default rule — `state === 'confirmed' && timeSinceUpdate === 0` — is what
   * ByteTrack / OC-SORT / BoT-SORT all use; `SortTracker` overrides this to
   * honor the original paper's warmup rule (output tentative tracks during
   * the first `minHits` frames).
   */
  protected exportConfirmed(): Track<TPayload>[] {
    const out: Track<TPayload>[] = [];
    for (const track of this.tracks.values()) {
      if (track.state === 'confirmed' && track.timeSinceUpdate === 0) {
        out.push(this.materializeTrack(track));
      }
    }
    return out;
  }

  /** Build a public {@link Track} from an {@link InternalTrack}. */
  protected materializeTrack(track: InternalTrack<TPayload>): Track<TPayload> {
    const det = track.lastDetection;
    const out: Track<TPayload> = {
      bbox: track.bbox,
      score: det.score,
      id: track.id,
      age: track.age,
      hits: track.hits,
      timeSinceUpdate: track.timeSinceUpdate,
      state: track.state,
    };
    if (det.classId !== undefined) out.classId = det.classId;
    if (det.payload !== undefined) out.payload = det.payload;
    return out;
  }
}
