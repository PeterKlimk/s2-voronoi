# Single-Thread Headroom Research Map

Status: compact research ledger, 2026-06-18.

This file is the canonical map for algorithmic single-thread headroom. It
replaces the older experiment diary shape: keep conclusions, branch names, and
enough numbers to avoid retreading. Full experiment transcripts live in git
history.

## Current Read

The remaining independence-safe prize is still mostly in completeness
certificates and candidate production, not in edge handoff or clip ordering.
However, the first directional certificate behavior branch has now been built
and is parked rather than "next to try."

- The design already harvests the big shared-edge win through same-bin
  edgechecks. Audits found no same-bin coverage leak.
- Edge-passed information contains real geometric signal, but every seed-order
  or anchored-splice behavior tried so far moves work earlier more than it
  removes work.
- Candidate-set locality exists, but previous-cell prefixes are not strong
  enough to skip packed selection or candidate production.
- Distance symmetry, as "known neighbor distance seeds an unseen bound", did
  not fire even under aggressive audits.
- Directional shell/cap certificates have real value in dense rebuilt regimes
  and are roughly neutral on fib, but the current implementation family is
  default-off/parked pending broader validation and merge judgment.
- Punch 1 is no longer just a design target: the axis-sort dense center-cell
  band prune exists and should be treated as the dense-regime baseline when
  interpreting certificate wins.

## Fade2D Baseline

We compared planar `compute_plane` against Fade2D v2.17.3 on identical uniform
inputs. The harness and Fade SDK were not retained; only the result matters for
strategy.

| n | compute_plane MT | Fade2D MT | compute_plane ST | Fade2D ST |
|---|---:|---:|---:|---:|
| 25k | 10.2 ms | 30.5 ms | 31.9 ms | 11.6 ms |
| 50k | 15.7 ms | 37.8 ms | 60.8 ms | 24.2 ms |
| 100k | 29.4 ms | 52.1 ms | 118.6 ms | 51.1 ms |

Reading:

- Fade wins ST by about 2.3x because incremental Delaunay touches real edges
  only, while this algorithm examines and rejects extra candidates to prove
  completeness.
- Our structural edge is MT scaling: independent cells parallelize better than
  a shared mutable triangulation.
- Honest independence-safe ST headroom is likely about 1.1-1.25x, concentrated
  in reducing examine-and-reject work.

## Active Or Promising

| idea | status | next action |
|---|---|---|
| Directional shell/cap certificate | parked promising | Branches `agent/directional-certificates` (`f51523a`) and `agent/directional-cell-cap-gate-audit` (`3eaf1c4`) showed the expected shape: roughly equal on fib, faster in mega/dense rebuilt regimes. Keep default-off until broader validation/product decision. |
| Candidate-production certificate work | open, narrower than before | New work must avoid candidate examination, or reuse already-paid metadata. Do not restart late known-batch support probing without a cheaper trigger. |
| Cheap conservative reject before clip | still open, lower priority | Only worth another try if the test is cheaper than a clip and can run without inflating branches on common unchanged candidates. |
| Angular-sweep clipper | speculative | Demoted: same-bin handoff already removes much of the duplicate fresh clipping this would optimize. Consider only after certificate/candidate work stalls. |

## Closed Or Parked

| branch | commit | verdict | takeaway |
|---|---|---|---|
| `agent/handoff-coverage-audit` | `f72167e` | recorded audit | Same-bin edgecheck handoff coverage was exact in fib/splittable/mega 100k. Do not chase a missing handoff leak without new evidence. |
| `agent/edgecheck-endpoint-seeds` | `e8eb6d5` | parked negative | Endpoint thirds are real changed clips, but broad early seeding costs more than it saves. |
| `agent/two-sided-edgecheck-seeds` | `fcc8b6e` | parked negative | Two-sided filtering is more selective, but still moves work earlier rather than removing it. |
| `agent/anchored-edge-witness-audit` | `0476781` | recorded signal | Two-sided endpoint witnesses match final edges. The math signal is real. |
| `agent/anchored-edge-clip-prototype` | `b1ef084` | parked negative/proof | Active anchored splicing is sound in tests but instruction-negative; it needs near-free synthesis or an upstream packed-production win. |
| `agent/edgecheck-early-cert-prototype` | `c828e35` | parked negative | Skipping packed for edgecheck-rich cells loses packed ordering and explodes shell work. |
| `agent/edgecheck-packed-seed-filter` | `c163474` | parked negative | Direct seed filtering is redundant with directed packed; endpoint removal before proof is unsound. |
| `agent/edgecheck-prebatch-fast-bound` | `d9e34a5` | parked near-neutral | Best pre-batch implementation, still slightly instruction-negative versus default off. |
| `agent/distance-symmetry-cert-prototype` | `69c6908` | recorded negative audit | Incoming edgecheck neighbor distances are not useful termination bounds; even aggressive farthest-seed bounds never terminate. |
| `agent/candidate-set-reuse` | `82e6ff8` | parked negative | Pre-clipping previous-cell edge neighbors increases work; candidate reuse must remove production/select work, not add seed clips. |
| `agent/candidate-production-overlap-probe` | `e38e016` | recorded signal | Loose adjacent-candidate overlap exists, but it was not yet a behavior win. |
| `agent/candidate-prefix-coverage-probe` | `cdc7527` | recorded signal | Small-prefix overlap is partial and regime-sensitive. |
| `agent/candidate-prefix-seed-prototype` | `684a722` | parked negative | Tiny seed lane reduced later stream hits but increased instructions. |
| `agent/packed-select-reuse-audit` | `e895f1e` | recorded negative audit | Previous packed first-prefix overlap is too sparse to skip selection, especially in dense regimes. |
| `agent/clip-conservative-reject-prototype` | `2d9f5bb` | parked negative/near-neutral | Tail-radius reject was near-neutral but instruction-negative; support reject was clearly worse. |
| `agent/frontier-prune-prototype` | `5e835da` | parked negative | Scalar shell-layer bound had zero skip hits in fib/splittable/mega 100k. |
| `agent/directional-certificates` | `f51523a` | parked promising | Integrated directional shell skip into the frontier path. Measurement read roughly equal on fib and faster in mega; keep as default-off research/productization candidate. |
| `agent/directional-cell-cap-gate-audit` | `3eaf1c4` | parked promising | Auto-gated shell-cell cap skip preserved fib/splittable and reduced mega 100k instructions by about 11.9%. |

## Key Measurements To Remember

Examine-and-reject ratio, fixed seed, ST, no preprocess:

| distribution | neighbors/cell | final edges/cell | examine/edge |
|---|---:|---:|---:|
| fib | 8.60 | 6.00 | 1.434 |
| uniform | 9.89 | 6.00 | 1.648 |
| gradient k=4 | 9.92 | 6.00 | 1.653 |
| splittable | 14.32 | 6.00 | 2.387 |

Known-batch directional shadow, timing-only lower bound:

| distribution | saved / neighbors |
|---|---:|
| fib | 30.6% |
| uniform | 32.5% |
| gradient k=4 | 32.5% |
| splittable | 34.6% |

Support-envelope shadow recovered 75-85% of exact shadow hits with zero false
positives in the timing probe, but the real late-batch behavior was too costly.

Shell-cell cap auto gate, branch `3eaf1c4`, non-timing perf counters:

| distribution / mode | instructions | cycles |
|---|---:|---:|
| fib auto on | 999.151M | 587.042M |
| fib auto disabled | 999.156M | 580.902M |
| splittable auto on | 2.847B | 1.423B |
| splittable auto disabled | 2.847B | 1.414B |
| mega f=0.8 auto on | 7.246B | 3.026B |
| mega f=0.8 auto disabled | 8.228B | 3.340B |

## Do Not Retread Without New Evidence

- More endpoint/anchored seed ordering tweaks. The signal is real, but the
  implementation family has repeatedly added more work than it removes.
- Packed bypass for seeded/edgecheck-rich cells. Packed order is essential.
- Direct packed filtering of edgecheck seeds. Directed packed already omits
  same-bin earlier locals.
- Previous-cell packed prefix reuse as a selection shortcut. Full-prefix hits
  are too rare.
- Distance-symmetry bound seeding with incoming edgecheck neighbor dots.
- Late known-batch directional support probes without a much cheaper trigger.
- A fresh "directional certificate" branch from scratch. The frontier-path
  version already exists; resume from `agent/directional-certificates` or the
  gated cap variant if productizing it.

## Cross-References

- `docs/optimization-ideas.md`: compact performance ledger and non-algorithmic
  optimization index.
- `docs/multi-regime-perf.md`: dense/sparse regime framing.
- `docs/micro-optimization-matrix.md`: paired micro-optimization evidence.
- `docs/dense-cell-subindex-design.md`: local dense-cell index design summary.
- `docs/punch1-center-cell-integration.md`: implemented Punch 1 center-cell
  band-prune details.

## One-Line Summary

Do not chase ST parity with Delaunay. Preserve independent cells, harvest the
small certificate/candidate-production wins, and keep dense-regime fixes
gated so uniform performance stays best-in-class.
