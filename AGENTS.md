# vestige.js — agent guidance

Notes for autonomous coding agents (Claude Code, Cursor, Aider, etc.) working in this
repo. For the general human-and-agent project conventions, see `CONTRIBUTING.md`.
`ARCHITECTURE.md` is the design spec.

This file is the thin overlay on top of those: instructions that are specifically about
how agents should structure their work, where to look, and what to cite.

---

## 1. The TDD scaffold-then-implement workflow

For each new layer of vestige.js — geometry primitives, Hungarian solver, Kalman filter,
BaseTracker, individual trackers (SORT/ByteTrack/OC-SORT/BoT-SORT), plugin interfaces —
do both the scaffold (type contracts, function signatures, JSDoc, comprehensive failing
tests) and the implementation. The maintainer reviews the PR before squash-merge.

The test-first cadence matters: the scaffold lands as its own commit on the feature
branch (CI red), then the implementation lands as the following commit(s) on the same
branch (CI green). The red → green arc on the branch is the contract — it shows the tests
were written against the spec, not against the implementation. See `CONTRIBUTING.md` §1
for the full branching / squash-merge convention.

**How to apply:**

- For a new layer, default to: feature branch → scaffold commit (tests + stubs, branch
  red) → implementation commit(s) (branch green) → open the PR.
- Order new functions foundational → derived so the test loop turns green in a clean
  cascade. A layer's tests only assume primitives that already have a green implementation
  on `main` (or earlier in the same branch).
- Hand-verify test oracle values where realistic. Commit pre-computed fixtures
  (scipy-generated, with the script that produced them) only when the function is too
  complex to verify by hand (e.g. Kalman update on an 8-d state) — see `ARCHITECTURE.md`
  §13.3 and `docs/decisions/0002-fixtures-layout.md`.
- Stubs use the `_paramName: T` underscore-prefix convention to mark intentionally-unused
  parameters. The codebase doesn't yet enable `noUnusedParameters`, but the convention
  pre-positions for it.
- **Throw-asserting tests must match an error message specific enough to fail against
  the `Error('not implemented')` stub** — otherwise the test trivially passes during
  the scaffold phase. Use patterns like `.toThrow(/positive[\s-]?definite/i)`, not
  `.toThrow()` or `.toThrow(/error/i)`.
- Before opening the PR, self-check the implementation against `CONTRIBUTING.md` §3
  (style) and §4 (reference checking). Cross-check non-obvious math against the Python
  reference listed in the §4.2 table and call out any deltas in the PR description.

## 2. Referencing project conventions in PRs and comments

- **PR descriptions and code comments should cite `CONTRIBUTING.md`, `ARCHITECTURE.md`,
  or the relevant `docs/decisions/NNNN-*.md` ADR.** All three are committed and resolvable
  by any reader with the repo. Do not invent references to other agent-local files.
- If a rule belongs in agent-only guidance, it lives here in `AGENTS.md` and is **not**
  cited from PRs or source comments (agents read this file; humans shouldn't have to).
- If you find yourself wanting a rule to live somewhere PRs can cite, promote it to
  `CONTRIBUTING.md` (workflow/style) or `ARCHITECTURE.md` (design-level).

## 3. Review style

When asked to review a diff (own or otherwise), be **specific about which
`CONTRIBUTING.md` §3 / §4 rule was missed** and **point to the file:line**. Generic
"this could be cleaner" feedback is not useful.
