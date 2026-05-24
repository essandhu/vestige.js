<!--
Thanks for contributing to vestige.js! Please fill out the sections that apply.
Delete sections that don't (e.g., if this is a docs-only PR, drop the benchmark block).
-->

## Summary

<!-- One or two sentences on what this PR changes and why. -->

## Type of change

<!-- Check all that apply. -->

- [ ] New tracker / algorithm
- [ ] New plugin implementation (motion predictor, cost function, embedder, CMC)
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Test / benchmark / fixture
- [ ] Documentation
- [ ] Build / tooling / CI
- [ ] Refactor (no behavior change)

## Linked references

<!--
For algorithm work, cite the paper / official implementation you're tracking.
For bug fixes, link the failing test or issue. Skip for docs/tooling PRs.
-->

- Paper / spec:
- Reference implementation:
- ARCHITECTURE.md section:

## Test plan

<!--
What new tests cover this change, and what manual checks did you run?
For algorithm PRs, include enough detail for a reviewer to reproduce.
-->

- [ ] Unit tests added / updated
- [ ] Property-based tests added / updated (where applicable)
- [ ] Snapshot tests added / updated (where applicable)
- [ ] `pnpm check` clean
- [ ] `pnpm typecheck` clean
- [ ] `pnpm test` green locally

## Benchmark / accuracy impact

<!--
Required for any change touching tracker logic, cost matrices, the Kalman filter,
or the Hungarian solver. Skip otherwise.
-->

- [ ] N/A — change does not affect tracking output
- [ ] HOTA / MOTA / IDF1 within the acceptance window vs. the published reference (ARCHITECTURE.md §10.4)
- [ ] Throughput regression < 5% on the standard benchmark scenarios
- [ ] Snapshot regression tests intentionally re-baselined (explain below)

<!-- Numbers / explanation here if relevant. -->

## Breaking changes

<!--
List any public API changes (renamed exports, changed signatures, removed options).
Default to "none" for v0.x unless you're explicitly proposing a break.
-->

- [ ] None
- [ ] Yes — described below

## Additional notes

<!-- Anything a reviewer should know that doesn't fit above. -->
