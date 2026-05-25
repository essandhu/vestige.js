# 0003 — Per-track lifecycle bookkeeping: exactly-once invariants for multi-stage trackers

- **Date:** 2026-05-24
- **Status:** Accepted
- **Scope:** Rules every concrete `BaseTracker` subclass must follow when calling `applyMatch` / `applyMiss` / `spawnTrack` / `sweepRemoved`. Motivated by a real bug landed in the initial ByteTracker implementation and caught in code review. Establishes a regression-test shape every new tracker (OC-SORT, BoT-SORT, …) must include.

## Context

The numerical-core ADRs (0001, 0002) cover correctness for primitives that are pure functions over flat arrays — geometry, IoU, linalg, Hungarian. Trackers are different: they hold mutable per-frame state in a `Map<number, InternalTrack>` and orchestrate that state through multi-stage matching pipelines. The first multi-stage tracker (`ByteTracker`, PR #10) shipped with a subtle bookkeeping bug that the unit and property tests did not catch:

> A confirmed track that missed both stage 1 (no high-score detection) and stage 2 (no low-score detection) had `applyMiss` called **twice** in one frame. Net effect: `timeSinceUpdate` advanced by 2 per missed frame, halving the user-facing `trackBuffer` retention.

The structural shape of the bug:

```
loop A:  if (cond) { track.state = 'lost'; applyMiss(track); }
loop B:  if (track.state === 'lost') applyMiss(track);
```

Loops A and B iterate overlapping snapshots of the same `InternalTrack` pool. Loop A mutates the field that loop B filters on. The track gets touched twice.

The same pattern is structurally available to every multi-stage tracker. OC-SORT will have three stages (ORU re-update + main + OCR); BoT-SORT will have the ByteTrack three-stage flow plus CMC. Both will have multiple `applyMiss` call sites per frame, and both will re-introduce this bug class unless we codify the rule now.

This ADR does not propose a code change. It captures the invariant the code already satisfies (after the fix in `eec3f9a`) and the test shape that catches violations.

---

## 1. The exactly-once invariant

For every track `t` present in `this.tracks` at the start of a frame, exactly one of the following must happen during that frame's `update()`:

| Outcome for `t` this frame | Required bookkeeping |
|---|---|
| Matched to a detection | `applyMatch(t, det)` called **once** |
| Unmatched | `applyMiss(t)` called **once** |
| Transitioned directly to `removed` without scoring (e.g. ByteTrack stage-3-unmatched tentatives) | Neither, by design — but document the case in code |

In all three cases, `t` is touched by **at most one** of `applyMatch` / `applyMiss` per frame. Never both. Never `applyMiss` twice.

**Why this matters:**

- `applyMiss` increments `timeSinceUpdate`. Calling it twice doubles the per-frame advance, which silently shortens the retention horizon (the bug above).
- `applyMatch` increments `hits` and `hitStreak`. Calling it twice double-counts hits, prematurely confirms tentative tracks, and violates the invariant `hits ≥ hitStreak`.
- The exported `Track.timeSinceUpdate` is the user's contract for "how long has this track been unobserved?" Doubled increments lie to the user.

For tracks **spawned this frame** the invariant doesn't apply — they were created mid-frame and start with `hits = 0`, `hitStreak = 0`, `tsu = 0` from `initTrack`. Tracker-specific post-spawn transitions (ByteTrack's frame-1 instant activation, OC-SORT's ORU on lost re-association) are allowed to mutate the fresh track's state, but should not invoke `applyMatch` on a track that has not actually been matched to a detection.

---

## 2. The state-mutation-then-state-filter anti-pattern

The bug shape that violates §1 in practice:

```ts
// Loop A: transition some tracks to 'lost'.
for (const ti of unmatchedConfirmed) {
  const t = tracks[ti];
  t.state = 'lost';
  this.applyMiss(t);
}
// Loop B: bump tsu for tracks already lost at frame start.
for (const ti of stage1Unmatched) {
  const t = tracks[ti];
  if (t.state === 'lost') this.applyMiss(t);   // ← matches BOTH cohorts
}
```

When the index sets overlap (`unmatchedConfirmed ⊂ stage1Unmatched`, which is typical for multi-stage matching where stage 2 operates on the stage-1-unmatched pool), the filter in loop B matches both:

- tracks that were lost **before** the frame began (intended), AND
- tracks that loop A just transitioned to `'lost'` (unintended — already bookkept by loop A).

**Two ways to avoid the trap:**

### 2a. Reorder so the filter loop runs before the mutation loop

This is what `ByteTracker` does after the fix:

```ts
// Loop B FIRST: only tracks already 'lost' at frame start match.
for (const ti of stage1Unmatched) {
  const t = tracks[ti];
  if (t.state === 'lost') this.applyMiss(t);
}
// Loop A SECOND: transition stage-2-fall-through confirmed → lost.
for (const ti of unmatchedConfirmed) {
  const t = tracks[ti];
  t.state = 'lost';
  this.applyMiss(t);
}
```

Cheapest fix. Works whenever the loops can be reordered without breaking other invariants.

### 2b. Snapshot the eligible set before any mutation

```ts
// Snapshot which stage1-unmatched tracks were already lost AT FRAME START.
const wereLostAtStart = new Set<number>();
for (const ti of stage1Unmatched) {
  if (tracks[ti].state === 'lost') wereLostAtStart.add(ti);
}
// ... mutations may happen freely now ...
for (const ti of stage1Unmatched) {
  if (wereLostAtStart.has(ti)) this.applyMiss(tracks[ti]);
}
```

Needed when loop ordering is constrained by a different invariant (e.g. matching results from loop A drive loop B's logic, not just filtering).

**Which to prefer:** §2a when possible. The reordering is local and zero-allocation. §2b is the escape hatch when reordering would break another rule (e.g. the cascade in `runStandardLifecycle` step 5, which deliberately reads its own just-set state to chain transitions).

---

## 3. Concrete guidance for new trackers

Every multi-stage tracker landing in this repo must:

1. **Audit every `applyMatch` / `applyMiss` call site.** Trace each track's path through one frame for the three cases below. Confirm each track is touched exactly once.
   - A confirmed track unmatched in every stage.
   - A confirmed track matched in the last stage (e.g. ByteTrack stage 2; OC-SORT OCR).
   - A lost track matched in stage 1.
   - A lost track unmatched everywhere.
   - A tentative track unmatched in its single chance.

2. **Avoid the anti-pattern in §2 explicitly.** When a stage mutates `track.state` and a subsequent stage reads `track.state` as a filter, document the order in a comment and explain why the reader doesn't see the mutation. (Example: the post-fix comment block at `bytetrack.ts:331-336`.)

3. **Add the §4 regression test** to the tracker's unit-test file. It's ~15 lines and catches the entire bug class for that tracker. Non-negotiable.

4. **Skip `runStandardLifecycle` deliberately.** Multi-stage trackers will not call `BaseTracker.runStandardLifecycle` (it embodies the SORT-default transitions). They write their own `update()` orchestration on top of the four shared primitives (`applyMatch` / `applyMiss` / `spawnTrack` / `sweepRemoved`).

---

## 4. Required regression test shape

Every concrete tracker must have a unit test that proves `timeSinceUpdate` advances by exactly 1 per missed frame. The cleanest form uses a tight retention window so the bug reaps the track immediately:

```ts
it('advances timeSinceUpdate by exactly 1 per missed frame for a confirmed-then-lost track', () => {
  const t = new YourTracker({ /* minimal trackBuffer / maxAge, e.g. 1 or 2 */ });
  const d = det([0, 0, 100, 100], 0.9);
  // Get to confirmed.
  // ...
  t.update([]);                                       // miss 1
  expect(t.getLostTracks()).toHaveLength(1);
  expect(t.getLostTracks()[0]?.timeSinceUpdate).toBe(1);
  // Optional: re-match in the next frame and assert id is preserved.
});
```

**Why a tight retention window:** the bug doubles the per-frame increment. With `maxAge = 1`, the buggy `tsu = 2` reaps the track in-frame (`2 > 1`); a same-bbox re-association in the next frame gets a fresh id. With the correct `tsu = 1`, the track stays lost (`1 > 1` is false) and the re-association restores the original id. **Both** the explicit `toBe(1)` assertion AND the id-preserved re-match catch the bug; using both is cheap insurance.

**Why this generalizes:** the test asserts the invariant directly, independent of which stages a tracker has. It catches the bug for any multi-stage tracker that violates §1.

Existing examples:

- `packages/core/tests/unit/bytetrack.test.ts` — block `'ByteTracker — lost-track retention (trackBuffer)'`, test `'advances timeSinceUpdate by exactly 1 per missed frame for a confirmed-then-lost track'`.
- `packages/core/tests/unit/sort-tracker.test.ts` — block `'SortTracker — lifecycle on missed frames'`, parallel test for symmetry. SortTracker is single-stage and currently cannot exhibit the bug, but the test guards against future regressions if `runStandardLifecycle` ever grows a second `applyMiss` site.

In addition to the per-tracker unit test, each tracker's property-test file (`tests/property/{sort,bytetrack,ocsort}-tracker.property.test.ts`) carries a `fast-check`-driven generalization — `'timeSinceUpdate equals exactly N after N consecutive missed frames'` — that probes the invariant across N ∈ [1, 20] and random bbox shapes. The property formulation catches the same bug class as the unit test plus any edge cases the hand-picked retention window in the unit test happens to miss.

---

## 5. Why this isn't a typing rule

A natural reaction is "can we encode §1 in the type system?" — e.g. make `applyMatch` / `applyMiss` return a branded token that must be consumed once per track. We don't, for three reasons:

1. **The TS type system can't easily express "exactly once per track per frame."** Linear types or session types would, but TS has neither, and shimming them via phantom types would add complexity that obscures the code.
2. **Trackers are the place where mutation is intentional.** ARCHITECTURE.md §2.1 explicitly carves out "imperative shell" for trackers; fighting it with types contradicts the design.
3. **The cost of the bug is bounded.** Doubled `tsu` is observable, locally reproducible, and caught by a 15-line test. A debug-mode invariant checker in `BaseTracker` (asserting at most one `applyMatch`-or-`applyMiss` per track per frame in a `__DEBUG__` build) is the heaviest enforcement worth considering, and is deferred until either (a) the bug recurs once more, or (b) the test-shape rule is violated by a PR author who didn't read this ADR.

---

## 6. Things deliberately deferred

| Item | When to revisit |
|---|---|
| `__DEBUG__`-mode invariant checker in `BaseTracker` | If the bug recurs after this ADR is in place, or if a tracker PR omits the §4 test and reviewers want a compile-time net. |
| ByteTracker sister fixture vs. `byte_tracker.py` (analog of `ocsort-noahcao/` and `sort-abewley/`) | When the same `exportConfirmed` / lifecycle-bookkeeping class of bug needs the same level of defense for ByteTracker. The harness from the two existing sister fixtures transfers verbatim; only `gen.py` and the sequence design change. |

---

## 7. Open questions

1. **Should `applyMatch` / `applyMiss` themselves assert against double-call?** A `track.__touchedThisFrame__` field guarded behind `__DEBUG__` would catch any violation regardless of test coverage. Cost: one boolean per track, reset at top of each `update()`. Worth it? Defer until §6's "if the bug recurs" trigger.
2. **Stage-3-unmatched tentatives in ByteTracker** are transitioned `state = 'removed'` without `applyMiss` (`bytetrack.ts:353`). Currently dormant because `sweepRemoved` runs immediately after, but the invariant in §1 admits this as the third category ("transitioned directly to removed without scoring"). Documented but not enforced; revisit if a hook between mutation and sweep ever exists.
3. **Does the rule generalize beyond `applyMatch` / `applyMiss`?** `spawnTrack` is naturally once-per-detection. `sweepRemoved` is once-per-frame. If a future primitive lands (e.g. an OC-SORT ORU re-update), the same invariant should be checked when documenting it.
