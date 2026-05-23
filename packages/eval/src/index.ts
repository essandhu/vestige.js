// Evaluation harness — placeholder.
//
// Per ARCHITECTURE.md §10 / §3, this package will house:
//   - MOT17/MOT20 sequence loaders (motchallenge/)
//   - HOTA, MOTA/MOTP, IDF1 metric implementations (metrics/)
//   - A benchmark runner that produces MOTChallenge-format JSON results.
//
// It is intentionally a separate package so it can use Node-only dev tooling
// (file I/O, CSV) without polluting the zero-dependency `vestige.js` bundle.

export {};
