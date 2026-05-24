import type { KalmanState } from '../filters/kalman.js';
import type { BBox, Detection, Track, Tracker, TrackState } from '../types.js';

/**
 * Per-frame book-keeping for a single track inside a {@link BaseTracker}.
 *
 * The library never exposes this shape directly — {@link BaseTracker.update}
 * materializes a public {@link Track} from it. The fields are mutated in place
 * by the lifecycle methods (`predictTrack`, `updateTrack`, lifecycle transitions);
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
 * and `associableTracks` arrays passed to {@link BaseTracker.associate} — not
 * to track IDs or to the full `tracks` map.
 *
 * Every detection index appears exactly once across `matched[*][1]` and
 * `unmatchedDetections`; every track index appears exactly once across
 * `matched[*][0]` and `unmatchedTracks`. The lifecycle relies on this
 * invariant when materializing the next frame.
 */
export interface AssociationResult {
  /** Matched pairs `[trackIndex, detectionIndex]`. */
  readonly matched: ReadonlyArray<readonly [number, number]>;
  /** Track indices (into `associableTracks`) that did not match a detection. */
  readonly unmatchedTracks: ReadonlyArray<number>;
  /** Detection indices (into `detections`) that did not match a track. */
  readonly unmatchedDetections: ReadonlyArray<number>;
}

/**
 * Lifecycle hyperparameters shared by every concrete tracker. See
 * ARCHITECTURE.md §6.1 for the state-machine semantics; defaults are not
 * provided here because each tracker (SortTracker, ByteTracker, ...) has its
 * own paper-specified defaults (ARCHITECTURE.md §§6.2–6.5).
 */
export interface BaseTrackerOptions {
  /**
   * Consecutive matches required to promote a tentative track to confirmed.
   * Must be ≥ 1.
   */
  readonly minHits: number;
  /**
   * Frames a `lost` track is retained before transitioning to `removed`.
   * `timeSinceUpdate > maxAge` triggers removal. Must be ≥ 0.
   */
  readonly maxAge: number;
}

/**
 * Shared per-frame lifecycle for every tracker in the SORT / ByteTrack /
 * OC-SORT / BoT-SORT family. See ARCHITECTURE.md §6.1.
 *
 * The template-method shape of {@link update} is the contract:
 *
 * 1. `frameIndex++`
 * 2. `predictTrack` on every non-removed track (subclass: motion model predict)
 * 3. `associate` the predicted tracks against this frame's detections (subclass)
 * 4. `updateTrack` on each matched pair (subclass: motion model update)
 * 5. Increment `timeSinceUpdate` on each unmatched track
 * 6. `initTrack` to spawn a new tentative track per unmatched detection (subclass)
 * 7. Advance lifecycle: tentative → confirmed / removed; confirmed → lost;
 *    lost → removed
 * 8. Sweep removed tracks out of the map
 * 9. Materialize and return the user-facing `Track[]` via {@link exportConfirmed}
 *
 * Subclasses customize steps 2, 3, 4, and 6 (and optionally 9). They never
 * touch the lifecycle transitions — those are the part the architecture says
 * is deliberately not pluggable (§8.4).
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

  protected constructor(protected readonly options: BaseTrackerOptions) {}

  /** Current frame counter; 0 before any call to {@link update}, then advances by 1 per call. */
  get frameIndex(): number {
    return this._frameIndex;
  }

  /**
   * Process one frame of detections and return the publicly-visible tracks.
   * Concrete; subclasses override the abstract motion + association hooks
   * rather than this method itself. See class docstring for the step ordering.
   */
  update(detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[] {
    this._frameIndex++;

    // 1. Predict every existing track forward one frame. age++ here so that
    //    tracks created later this frame (in step 4) correctly start at age 0.
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
    const { matched, unmatchedTracks, unmatchedDetections } = this.associate(
      detections,
      associable,
    );

    // 4a. Update matched tracks; a lost track that re-matches goes back to confirmed.
    for (const [ti, di] of matched) {
      const track = associable[ti];
      const det = detections[di];
      if (!track || !det) continue;
      this.updateTrack(track, det);
      track.hits++;
      track.hitStreak++;
      track.timeSinceUpdate = 0;
      if (track.state === 'lost') track.state = 'confirmed';
    }

    // 4b. Mark unmatched tracks: hitStreak resets, timeSinceUpdate advances.
    for (const ti of unmatchedTracks) {
      const track = associable[ti];
      if (!track) continue;
      track.hitStreak = 0;
      track.timeSinceUpdate++;
    }

    // 4c. Spawn a new tentative track per unmatched detection. IDs are assigned
    //     here so that subclasses' initTrack doesn't need to know about nextId.
    for (const di of unmatchedDetections) {
      const det = detections[di];
      if (!det) continue;
      const fresh = this.initTrack(det);
      fresh.id = this.nextId++;
      this.tracks.set(fresh.id, fresh);
    }

    // 5. Advance lifecycle states. The checks chain (not `else if`) so that a
    //    confirmed track that just missed on a frame with maxAge = 0 cascades
    //    confirmed → lost → removed in a single pass — matching sort.py's
    //    `if (trk.time_since_update > self.max_age): pop` semantics. Tentative
    //    tracks follow the same removal rule: they survive `maxAge` consecutive
    //    misses, just like confirmed tracks. (sort.py has no separate tentative
    //    state — every tracker is reaped on `tsu > max_age` regardless of how
    //    many times it matched.)
    for (const track of this.tracks.values()) {
      if (track.state === 'tentative') {
        if (track.timeSinceUpdate > this.options.maxAge) {
          track.state = 'removed';
        } else if (track.hitStreak >= this.options.minHits) {
          track.state = 'confirmed';
        }
      }
      if (track.state === 'confirmed' && track.timeSinceUpdate > 0) {
        track.state = 'lost';
      }
      if (track.state === 'lost' && track.timeSinceUpdate > this.options.maxAge) {
        track.state = 'removed';
      }
    }

    // 6. Sweep removed tracks out of the map.
    for (const [id, track] of this.tracks) {
      if (track.state === 'removed') this.tracks.delete(id);
    }

    // 7. Export.
    return this.exportConfirmed();
  }

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
   * unmatched, still within `maxAge`).
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
   * motion model's predict step (e.g. {@link import('../filters/kalman.js').KalmanFilter.predict})
   * and refresh `track.kalmanState` *and* `track.bbox` in place.
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
   * track must have `id = 0` (caller assigns the real id), `age = 0`,
   * `hits = 0`, `hitStreak = 0`, `timeSinceUpdate = 0`, and `state = 'tentative'`.
   * `kalmanState` and `bbox` are initialized from `detection`.
   *
   * The `hits = 0` convention matches `abewley/sort` (`sort.py:KalmanBoxTracker.__init__`):
   * the spawn detection is **not** counted as a hit. The first hit is recorded
   * the next time the track matches a detection, via {@link updateTrack} and the
   * lifecycle in {@link update}. Subclasses that diverge from this convention
   * (e.g. counting the init detection as a hit) will confirm tracks one frame
   * earlier than the reference implementation.
   */
  protected abstract initTrack(detection: Detection<TPayload>): InternalTrack<TPayload>;

  /**
   * Associate the predicted tracks against this frame's detections. Indices
   * in the returned {@link AssociationResult} refer to the `associableTracks`
   * and `detections` arrays passed in. See class docstring for the broader
   * lifecycle contract.
   */
  protected abstract associate(
    detections: ReadonlyArray<Detection<TPayload>>,
    associableTracks: ReadonlyArray<InternalTrack<TPayload>>,
  ): AssociationResult;

  /**
   * Materialize the public {@link Track[]} returned from {@link update}. The
   * default rule — `state === 'confirmed' && timeSinceUpdate === 0` — is what
   * ByteTrack / OC-SORT / BoT-SORT all use; SortTracker overrides this to
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
