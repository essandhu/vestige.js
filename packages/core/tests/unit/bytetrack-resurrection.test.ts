/**
 * Pins vestige's chosen behavior at the `max_time_lost` boundary, where it
 * DELIBERATELY diverges from `byte_tracker.py`.
 *
 * The reference has a one-frame track-resurrection bug. In `byte_tracker.py:283-284`
 * the per-frame bookkeeping runs
 *
 *     self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
 *     self.removed_stracks.extend(removed_stracks)
 *
 * — the subtraction uses the ACCUMULATED removed list, which does not yet contain
 * the tracks removed on *this* frame; those are appended on the next line. So a
 * track marked removed on frame N survives in `lost_stracks` through frame N+1,
 * stays in that frame's `strack_pool`, and if its detection reappears right then,
 * stage 1 `re_activate()`s it under its ORIGINAL id. (The same object ends up in
 * `tracked_stracks` and `removed_stracks` simultaneously — it is unambiguously a
 * bug, not a design choice.)
 *
 * Net effect: the reference gives a lost track re-association chances on frames
 * L+1..L+32; vestige gives it L+1..L+31. We do not reproduce the extra frame —
 * see ADR-0003 §7 item 1. Reproducing it would violate the ADR's "removed means
 * removed" invariant to chase a reference defect.
 *
 * This test is what makes that a decision rather than an accident: if someone
 * later "fixes" the off-by-one to match the reference, this fails and points them
 * at the ADR.
 */
import { describe, expect, it } from 'vitest';
import { ByteTracker } from '../../src/trackers/bytetrack.js';
import type { BBox, Detection } from '../../src/types.js';

const BOX: BBox = [10, 10, 50, 60];
const det = (): Detection => ({ bbox: BOX, score: 0.9 });

/**
 * Drive a static object for `L` frames, then withhold it for `gap` frames, then
 * bring it back. Returns the ids exported on the first frame it reappears and on
 * the frame after (a track re-spawned from scratch is tentative on its first frame
 * and only exported once stage 3 confirms it, so both frames matter).
 */
function runGap(gap: number, trackBuffer?: number): { onReturn: number[]; afterReturn: number[] } {
  // Default maxAge = floor(30 / 30 * 30) = 30.
  const tracker = trackBuffer === undefined ? new ByteTracker() : new ByteTracker({ trackBuffer });
  for (let i = 0; i < 5; i++) tracker.update([det()]); // born frame 1, matched through frame 5
  for (let i = 0; i < gap; i++) tracker.update([]);
  const onReturn = tracker.update([det()]).map((t) => t.id);
  const afterReturn = tracker.update([det()]).map((t) => t.id);
  return { onReturn, afterReturn };
}

describe('ByteTracker — max_time_lost boundary (deliberate divergence from the reference)', () => {
  it('re-associates under the ORIGINAL id after exactly maxAge (30) missed frames', () => {
    // tsu reaches 30 on the last missed frame; `30 > 30` is false, so the track is
    // still `lost` when the detection returns and stage 1 re-finds it. The reference
    // agrees (its `frame_id - end_frame = 31 > 30` reap has not fired yet either).
    const { onReturn } = runGap(30);
    expect(onReturn).toEqual([1]);
  });

  it('mints a NEW id after 31 missed frames, where the reference RESURRECTS the old one', () => {
    // THE DIVERGENCE, and the only gap length at which it shows.
    //
    // vestige: tsu hits 31 > 30 on the 31st missed frame -> 'removed' -> swept. When
    // the detection returns there is no track left, so it spawns a fresh tentative:
    // nothing is exported on the return frame, and a NEW id appears once stage 3
    // confirms it on the frame after.
    //
    // Reference: the track was marked Removed on that same frame but is STILL in
    // `lost_stracks` (byte_tracker.py:283-284), hence still in `strack_pool`, so the
    // returning detection re-activates it and it is exported immediately as id 1.
    //
    // Verified against the reference at the pinned commit: a 31-miss gap yields
    // `id=1` on the return frame; a 32-miss gap yields a new id. See ADR-0003 §7.
    const { onReturn, afterReturn } = runGap(31);
    expect(onReturn).toEqual([]); // reference would emit [1] here
    expect(afterReturn).toEqual([2]); // NEW id, not the reference's resurrected 1
  });

  it('mints a NEW id after 32 missed frames — past the resurrection window, both agree again', () => {
    // By now the reference has swept the track out of `lost_stracks` too, so both
    // implementations spawn a fresh track. This is the gap length the
    // bytetrack-foundationvision fixture uses for track E, which is why that fixture
    // tests the max_time_lost boundary rather than this bug.
    const { onReturn, afterReturn } = runGap(32);
    expect(onReturn).toEqual([]);
    expect(afterReturn).toEqual([2]);
  });
});

/**
 * The reap boundary at `maxAge = 0`, which is NOT the resurrection divergence above and
 * IS a bug we fix.
 *
 * The reference's removal loop (`byte_tracker.py:271-274`) iterates `self.lost_stracks` —
 * and the tracks that went lost on THIS frame are not in it yet. They sit in a local
 * `lost_stracks` list and are only folded in by `self.lost_stracks.extend(lost_stracks)`
 * on the line AFTER the loop. So a track is never eligible for removal on its first lost
 * frame; the check first sees it one frame later.
 *
 * At `maxAge >= 1` that lag is invisible — the track survives its first lost frame on the
 * age budget anyway. At `maxAge = 0` it is the whole behaviour: `max_time_lost = 0` ends
 * up acting exactly like 1. Executed against the reference at the pinned commit, the two
 * configs produce IDENTICAL output:
 *
 *     track_buffer=0        track_buffer=1
 *     gap=1: [1] / [1]      gap=1: [1] / [1]
 *     gap=2: [1] / [1]      gap=2: [1] / [1]
 *     gap=3: []  / [2]      gap=3: []  / [2]
 *
 * vestige reaped on `timeSinceUpdate > maxAge`, which at maxAge=0 fires on the very frame
 * the track goes lost (tsu = 1 > 0) — removing it a frame before the reference's loop can
 * even see it, and losing the re-association entirely. The fix is to reap on
 * `timeSinceUpdate > max(maxAge, 1)`, which encodes "the removal check cannot see a track
 * that went lost this frame" and is a no-op for every maxAge >= 1.
 *
 * See ADR-0003 §7.
 */
describe('ByteTracker — reap boundary at maxAge = 0 (the removal loop lags by one frame)', () => {
  it('still re-associates after a 1-frame gap, exactly as the reference does', () => {
    // THE BUG. vestige used to reap on the frame the track went lost, so the returning
    // detection found nothing and minted a new id.
    const { onReturn, afterReturn } = runGap(1, 0);
    expect(onReturn).toEqual([1]);
    expect(afterReturn).toEqual([1]);
  });

  it('mints a NEW id after a 2-frame gap — the sanctioned resurrection divergence', () => {
    // The reference recovers id 1 here, via the one-frame resurrection window documented
    // at the top of this file. We deliberately do not. Exactly ONE divergent gap length,
    // which is the same shape as every maxAge >= 1 config.
    const { onReturn, afterReturn } = runGap(2, 0);
    expect(onReturn).toEqual([]);
    expect(afterReturn).toEqual([2]);
  });

  it('mints a NEW id after a 3-frame gap, exactly as the reference does', () => {
    const { onReturn, afterReturn } = runGap(3, 0);
    expect(onReturn).toEqual([]);
    expect(afterReturn).toEqual([2]);
  });

  it('behaves identically to trackBuffer = 1, as the reference does', () => {
    // The reference's own output for track_buffer=0 and track_buffer=1 is byte-identical
    // (see the block comment). Anything else means the lag is not being modelled.
    for (const gap of [1, 2, 3, 4]) {
      expect(runGap(gap, 0), `gap=${gap}`).toEqual(runGap(gap, 1));
    }
  });
});
