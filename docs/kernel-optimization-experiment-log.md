# Kernel optimization experiment log

This log records the read-only review synthesis, measurement gates, and branch outcomes from the
July 2026 kernel pass. Wall-clock measurements taken while the shared machine is busy are
non-decisive. Retired instructions and branches are the primary behavioral signals; timing-only
counts are used for workload censuses.

## Pass closeout

The original multi-model shortlist is exhausted. The measurement infrastructure and read-only
oracles were retained; every production behavior experiment in that shortlist was rejected and
reverted. A later structural follow-up, group-shared shell traversal, passed its read-only gate and
is recorded separately below. The branch names in this table are archival experiment labels, not
outstanding merge candidates.

| Family | Branches | Final decision |
| --- | --- | --- |
| Fixed smaller packed prefix | `agent/kernel-demand-prefix` | Rejected: the 8-slot first batch saved 0.315% instructions on Fibonacci but added 2.777% on uniform and increased branches on both. |
| Compact high-key overflow | `agent/kernel-compact-overflow`, `agent/kernel-compact-top64-cost`, `agent/kernel-compact-overflow-rebuild` | Shadow census retained; heap-based additive and true-replacement forms rejected at +5.1--11.4% and +6.9--7.7% instructions respectively. |
| Shell-cell rejection | `agent/kernel-shell-cell-reject`, `agent/kernel-shell-cell-cap`, `agent/kernel-shell-cell-cap-skip` | Exact and conservative-cap oracles retained; the order-preserving production form was rejected at +1.8--3.5% instructions. |
| Center-informed high threshold | `agent/kernel-threshold-shadow`, `agent/kernel-threshold-one-shot` | Center-only prediction and per-ring-cell sampling were rejected first. The final exact-center pre-gate plus one-vector ring sample isolated clustered overshoot, but its margin collapsed with scale and on splittable controls; no probe code retained. |
| Seed-first packed preparation | `agent/kernel-seed-first-oracle` | Rejected with current metadata: the exact ceiling is only 3--6% of row dots, production visits one later candidate per exact-batch hit, and whole-cell caps retain at most 1.43% of row dots with negligible key savings. No probe code retained. |
| Same-cell regional local hull | `agent/kernel-regional-hull-oracle` | Rejected before replay: no measured same-grid-cell group repaid even the optimistic pair-count floor of the current naive local hull. The best case reached 74.4% before guard expansion, exact-predicate cost, certification, or stitching. No probe code retained. |
| Group-shared shell traversal | `agent/kernel-shared-shell-oracle` | Read-only gate passed: repeated group-local cell traversal is 93.5--99.1% redundant and resident/query work has high active width on every shell-using workload. Promoted to `PERF-003`; no probe code retained. |

The negative results share a useful conclusion: the existing width-one unchanged clip and
append-then-partition selection paths are already cheap. A successor should remove dot/key work
before those paths, change the cutoff without per-key maintenance, or use a regional algorithm
whose construction cost is materially below the current naive local hull. The closed structural
follow-ups are recorded in
[`algorithmic-performance-ideas.md`](algorithmic-performance-ideas.md#post-review-kernel-hypotheses);
`PERF-002` in [`work-log.md`](work-log.md#perf-002--post-review-kernel-hypotheses) records the
closure and reopening conditions. `PERF-003` records the distinct shared-shell successor.

## Common census baseline

- Branch: `agent/kernel-census`
- Commit: `22a1ea4` (`perf: add packed batch timing census`)
- Parent: `badfa64`
- Adds timing-feature-only emitted/visited/abandoned exact-batch counts, split into chunk0/tail and
  first/later batches, plus machine-readable packed selection timings.
- The ordinary `tools` release build has a byte-identical `.text` section to `badfa64`, so the
  census has zero production instruction impact.
- Validation: `cargo test --release --features timing --lib` and `cargo test --release` passed.

### Packed batch census

| Workload | Class | Emitted | Visited | Abandoned | Abandoned |
| --- | --- | ---: | ---: | ---: | ---: |
| 500k Fibonacci | chunk0 first | 5,737,473 | 2,115,115 | 3,622,358 | 63.1% |
| 500k Fibonacci | tail first | 39,956 | 9,569 | 30,387 | 76.1% |
| 500k uniform | chunk0 first | 5,838,059 | 3,223,654 | 2,614,405 | 44.8% |
| 500k uniform | chunk0 later | 134,152 | 43,812 | 90,340 | 67.3% |
| 500k uniform | tail first | 246,344 | 100,528 | 145,816 | 59.2% |
| 500k uniform | tail later | 29,521 | 11,691 | 17,830 | 60.4% |
| 100k clustered | chunk0 first | 1,351,073 | 788,457 | 562,616 | 41.6% |
| 100k clustered | chunk0 later | 968,719 | 807,035 | 161,684 | 16.7% |
| 100k clustered | tail first | 47,166 | 40,113 | 7,053 | 15.0% |
| 100k clustered | tail later | 746,116 | 733,567 | 12,549 | 1.7% |

The abandoned-slot premise for demand-sized prefixes is real on ordinary inputs. It is not by
itself proof that a smaller request saves work: a smaller ask can add another full-remainder
partition pass, and chunk0's small-remainder path may replace one whole sort with repeated
partitioning.

## Adversarial review decisions

All three blind adversarial reviews returned **measure-only**:

- Demand-sized prefix: runtime-sized extraction is locally sound, but request-aware frontier-cache
  contracts and a cheap deterministic demand policy are missing. Measure selection path and
  remainder shape; if abandonment is strong, test only a fixed 8-versus-16 probe first.
- Compact overflow: preserve the full `(descending dot, ascending slot)` key order and keep overflow
  inside chunk0 before tail. First simulate cap exceedance and marked-block density using existing
  exact keys. Dense-band queries require separate treatment.
- Shell-cell rejection: a gnomonic rejection can become invalid if a later candidate triggers
  spherical fallback. First run an exact layer-start resident-cell oracle; any implementation needs
  replay/restart or a proof that crosses representation changes.

## Demand-prefix branch

- Branch: `agent/kernel-demand-prefix`
- Experiment: change the fixed first packed batch from 16 to 8; no adaptive policy.
- Result: rejected and left uncommitted. Targeted ordering/bound/termination tests passed.

Seven interleaved native single-threaded `perf stat` pairs at 500k, no preprocessing:

| Workload | Instructions | Branches | Branch misses | Cycles |
| --- | ---: | ---: | ---: | ---: |
| Fibonacci | -0.315% (7/7) | +3.913% (0/7) | +13.23% (0/7) | -3.58% (5/7) |
| Uniform | +2.777% (0/7) | +7.435% (0/7) | +12.78% (0/7) | -0.55% (3/7) |

The uniform instruction and branch regressions decisively reject a globally smaller first batch.
Fibonacci confirms that useful saved work exists, but selecting when to request less must be
predictable from cheap, deterministic cell state and must avoid repeated partition scans.

Raw counter files for this workspace session:

- `/tmp/kernel-demand-prefix-fib.csv`
- `/tmp/kernel-demand-prefix-uniform.csv`

## Compact-overflow branch

- Branch: `agent/kernel-compact-overflow`
- Commit: `e147a93` (`perf: simulate compact packed overflow`)
- Timing-only shadow simulator; ordinary release `.text` remains byte-identical to the common
  census baseline.
- The simulator sorts each dead post-group exact key list by the production full `u64` order, then
  models a retained prefix plus absolute-slot blocks. It distinguishes all over-cap queries from
  the subset whose baseline frontier actually emitted beyond the cap.
- Validation: `cargo test --release --features timing --lib` passed.

100k clustered, normal packed queries only (no ready packed query used dense-band mode):

| Metric | cap 32, block 16 | cap 64, block 16 |
| --- | ---: | ---: |
| Ready packed queries | 96,433 | 96,433 |
| Queries over cap | 63,588 (65.9%) | 49,562 (51.4%) |
| Queries that emitted beyond cap | 5,556 (5.8%) | 2,570 (2.7%) |
| High keys | 17,098,541 | 17,098,541 |
| Keys beyond cap | 14,630,501 (85.6%) | 12,842,865 (75.1%) |
| Eligible slots for demanded rescans | 3,254,908 | 1,367,804 |
| Slots covered by cheap any-high blocks | 2,891,515 (88.8%) | 1,237,094 (90.4%) |
| Slots covered by exact deferred blocks | 2,806,536 (86.2%) | 1,186,506 (86.7%) |

This is mixed rather than an automatic promotion. A cap removes most retained keys and few queries
request reconstruction, but requested block masks are effectively dense: they revisit about 90%
of those queries' eligible neighborhoods. Cap 64 reduces absolute rescan work to 1.24M dots, over
an order of magnitude below total high-key materialization, but a production design must also pay
streaming top-64 maintenance on all 17.1M high keys. Do not build the behavioral version until a
cheap bounded-selection design has an instruction-cost model or isolated microbenchmark capable
of overcoming that eager cost.

### Cap-64 maintenance and rebuild follow-up

Two production-codegen experiments tested the unresolved eager cost. Both were exploratory,
rejected, reverted, and left uncommitted.

The first, on `agent/kernel-compact-top64-cost`, added a fixed-capacity max-heap of the best 64
full keys at every existing key-production site while retaining the baseline vectors and emissions.
The heap matched a fully sorted top-64 at every tested prefix, including ties and key extrema; all
packed ordering/bound tests remained unchanged. This is an intentionally conservative additive
cost ceiling because it does not claim the later storage/selection savings.

Seven pinned native single-threaded `perf stat` pairs, no preprocessing:

| Workload | Instructions | Branches | Branch misses | Cycles |
| --- | ---: | ---: | ---: | ---: |
| 500k Fibonacci | +5.114% | +8.238% | +47.78% | +12.79% mean |
| 100k clustered, seed 1 | +11.353% | +22.984% | +74.77% | +28.02% mean |

The second, on `agent/kernel-compact-overflow-rebuild`, replaced ordinary retention for non-band
queries rather than adding to it. It:

- kept the exact best 64 keys under the production `(descending dot, ascending slot)` order;
- recorded deduplicated absolute-slot block-16 source masks;
- exposed the retained worst-key dot as the exact unseen bound;
- rebuilt and fully ordered overflow only when the consumer advanced past that bound; and
- left dense-band queries on the baseline path.

A 256-way tied fixture forced the rebuild seam and reproduced the complete ascending-slot tie
order. Packed brute-force order/bound tests, the API suite, and the correctness suite passed.
Nevertheless, deleting overflow vector growth and unused partition/sort work did not repay the
streaming heap and block bookkeeping:

| Workload | Instructions | Branches | Branch misses | Cycles |
| --- | ---: | ---: | ---: | ---: |
| 500k Fibonacci | +6.914% | +15.712% | +55.84% | +14.45% |
| 100k clustered, seed 1 | +7.721% | +30.451% | +83.44% | +18.93% |

The retired-instruction and branch losses are decisive; wall clock was not used. The compact
overflow line is closed for this selection strategy. Its memory ceiling is real, but the current
append-then-partition kernel is substantially cheaper than exact streaming top-64 maintenance.
Revisit only if a future design avoids a data-dependent heap comparison/repair for every high key,
or if memory—not construction throughput—becomes the explicit objective.

Raw counter files for this workspace session:

- `/tmp/compact_top64_fib.csv`
- `/tmp/compact_top64_clustered.csv`
- `/tmp/compact_rebuild_fib.csv`
- `/tmp/compact_rebuild_clustered.csv`

## Exact shell-cell oracle branch

- Branch: `agent/kernel-shell-cell-reject`
- Commit: `d6c6bef` (`perf: add shell-cell rejection oracle`)
- Timing-only exact oracle at frozen shell-layer start. It groups emitted residents by grid cell,
  tests every not-yet-attempted resident against the current gnomonic polygon, and records the
  ideal cell/slot ceiling plus fallback-after-hit exposure.
- The ordinary `tools` release `.text` is byte-identical to the common census baseline.
- Validation: timing-feature library tests (271 passed, 5 ignored), timing-feature all-target
  clippy, and the shell grouping/one-shot snapshot test passed.

The useful class is `later_all`: later shell layers whose cells emit every resident. First-layer
center-directed cells expose only a directional subset, so a whole-cell certificate is naturally
too coarse there.

| Workload | Exact hit cells | Exact hit slots | Eligible slots | Exact hit slots |
| --- | ---: | ---: | ---: | ---: |
| 100k clustered, seed 1 | 97,403 / 102,472 | 3,959,094 | 4,170,867 | 94.9% |
| 500k uniform, seed 1 | 8,500 / 8,675 | 155,545 | 158,423 | 98.2% |
| 100k mega, default seed | 177,271 / 211,777 | 2,451,104 | 3,033,579 | 80.8% |

The upper bound is strong, but the exact per-resident predicate is not a candidate implementation:
clustered alone performs 3.73M exact predicate calls to find 3.96M hit slots. More importantly, the
default mega run records three later spherical handoffs with 5,244 previously certified slots
still unconsumed. Gnomonic monotonicity therefore does not authorize deleting a whole layer's
residents from the stream.

## Conservative shell-cell cap branch

- Branch: `agent/kernel-shell-cell-cap`
- Commit: `7587bed` (`perf: measure shell cell cap certificate`)
- Timing-only cell certificate. For each polygon vertex it minimizes the exact chord-bisector
  half-space value over the grid cell's conservative spherical cap and the global canonical-point
  norm envelope. It does not load individual residents.
- Differential comparison against the exact oracle reported zero false-positive cells/slots on
  clustered, uniform, and two mega seeds.
- The ordinary `tools` release `.text` remains byte-identical to `d6c6bef`.
- Validation: timing-feature library tests (271 passed, 5 ignored), timing-feature all-target
  clippy, and ordinary-release codegen comparison passed.

| Workload | Cap hit cells | Cap hit slots | Of exact hit slots | Of eligible slots |
| --- | ---: | ---: | ---: | ---: |
| 100k clustered, seed 1 | 75,368 | 3,300,214 | 83.4% | 79.1% |
| 500k uniform, seed 1 | 7,211 | 132,909 | 85.4% | 83.9% |
| 100k mega, default seed | 164,503 | 2,371,392 | 96.7% | 78.2% |

The fallback hazard survives the conservative filter: on default-seed mega, two handoffs occur
with 4,757 cap-certified slots still unconsumed. Any future pre-scan rejection must retain exact
logical ordering or supply a correct replay/restart mechanism.

### Order-preserving production prototype

A separate `agent/kernel-shell-cell-cap-skip` prototype retained every candidate in the existing
dot/sort stream and in its existing logical position. It bypassed only the gnomonic clip for a
cap-certified candidate and disabled all later bypasses immediately after spherical handoff. This
avoided the replay ambiguity, but it did not remove resident dot products, key construction, or
sorting. The exploratory changes were rejected and reverted without a commit after validation.

Seven interleaved pinned, native, single-threaded `perf stat` pairs, no preprocessing:

| Workload | Instructions | Branches | Branch misses | Cycles |
| --- | ---: | ---: | ---: | ---: |
| 100k clustered, seed 1 | +1.804% | +3.401% | +1.435% | -2.02% mean, noisy |
| 100k mega, default seed | +3.499% | +2.905% | +8.148% | +11.56% |

The deterministic instruction and branch losses reject this form. The existing unchanged clip is
already cheap (including its radial early rejection), so a polygon-versus-cap test plus per-layer
metadata and range membership costs more than bypassing the remaining clip plumbing. The mega
cycle result was adverse in all seven pairs (+4.1% to +18.7%), despite wall clock being otherwise
ignored on the busy host.

Raw counter files for this workspace session:

- `/tmp/shell_cap_clustered.csv`
- `/tmp/shell_cap_mega.csv`

Do not promote the cap certificate merely because its geometric coverage is high. It becomes
interesting again only as part of a design that removes earlier work—resident dot/key generation
or sorting—while preserving exact stream order across a possible later fallback. That is a
different, more structural experiment.

## Center-informed threshold shadow branch

- Branch: `agent/kernel-threshold-shadow`
- Experiment: timing-only simulation of a one-shot raised packed high threshold. Production
  thresholds, exact ordering, emitted batches, and fallback behavior were unchanged.
- Validation while the probe was present: `cargo test --release --features timing --lib` passed
  (273 passed, 5 ignored), including histogram boundary tests. The probe was removed after the
  decision so rejected instrumentation did not become permanent code burden.

The first form used only already-computed directed center-cell dots. It was decisively biased:
center residents are systematically nearer than ring residents, so the estimator raised thresholds
on ordinary controls whose baseline key count was already at the intended budget.

| 100k workload | Baseline keys | Predicted keys | Budget oracle | New tail visits | Rescan dots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clustered, seed 1 | 16,286,039 | 8,445,319 | 2,437,927 | 5,458 | 2,551,410 |
| Mega, default seed | 1,283,904 | 667,885 | 1,210,366 | 2,296 | 995,931 |
| Fibonacci | 1,281,679 | 496,173 | 1,281,662 | 22,733 | 2,697,862 |
| Uniform, seed 1 | 1,307,579 | 484,429 | 1,302,857 | 46,092 | 5,487,269 |

The Fibonacci and uniform rows reject center-only prediction: it invents roughly 0.8M of apparent
key savings where the exact 32-key budget oracle says essentially none exist, then forces tens of
thousands of queries across the new tail boundary.

A single refinement sampled up to eight actual residents from every eligible ring cell, maintained
per-query 64-bin normalized histograms, and limited the census to non-band center cells with at
least 64 residents. This fixed the control failure—Fibonacci and uniform were excluded by the
occupancy gate—and made the tail-crossing estimate conservative enough to expose a real clustered
ceiling:

| Workload | Baseline keys | Predicted keys | Ring sample dots | New tail visits | Rescan dots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100k clustered, seed 1 | 16,085,400 | 8,982,206 | 1,457,003 | 775 | 283,515 |
| 100k clustered, seed 2 | 15,058,067 | 8,928,951 | 1,562,101 | 783 | 309,504 |
| 100k clustered, seed 3 | 16,611,473 | 9,339,342 | 1,508,341 | 808 | 314,187 |
| 500k clustered, seed 1 | 35,421,070 | 17,156,691 | 8,145,817 | 2,934 | 1,212,254 |
| 100k splittable, seed 1 | 4,277,219 | 2,172,807 | 1,436,647 | 459 | 209,845 |
| 100k mega, default seed | 1,264,380 | 1,201,864 | 2,461,875 | 165 | 62,865 |
| 100k bimodal, seed 1 | 1,285,036 | 1,236,323 | 2,490,626 | 146 | 98,126 |
| 100k gradient, seed 1 | 930,603 | 920,816 | 2,059,448 | 121 | 87,473 |
| 100k outlier, seed 1 | 129,009 | 127,727 | 17,184 | 4 | 2,536 |

The refinement is not a production candidate as measured. On 100k clustered seed 1 it removes
7.10M keys for 1.46M sample dots plus 0.28M newly required rescan dots, but at 500k the ratio falls
to 18.26M keys for 9.36M extra dots. More importantly, the same occupancy gate spends 2.06--2.49M
sample dots to save only 0.01--0.06M keys on mega, bimodal, and gradient inputs. Grid rebuild state
does not separate the regimes: both useful and useless non-rebuilt cases exist.

The original center-only hypothesis is closed. The narrower exact-center pre-gate and one-vector
ring sample were subsequently measured and also closed; see the next section.

## One-shot threshold refinement oracle branch

- Branch: `agent/kernel-threshold-one-shot`
- Experiment: timing-only simulation after the directed center pass. Non-band queries were eligible
  only when their exact center high-key count already exceeded the 32-key budget. Each eligible
  query evaluated one evenly distributed eight-resident ring sample, estimated a raised threshold,
  and was tested at raw, 1x, 2x, 4x, 8x, and 16x predicted-saving margins.
- Cheap feasibility gate: before sampling, reject any query whose absolute maximum possible saving
  from the known center count and ring population could not meet the selected margin.
- Accounting: eager keys demoted on a query that would use the tail were classified as rebuilt,
  not saved. The oracle separately charged feasible sample-vector evaluations, selected center keys
  that a threshold selection must scan, newly induced tail requests, and their full rescan dots.
- Validation while present: timing-feature release library tests passed (271 passed, 5 ignored),
  including a weighted-budget estimator test; timing-feature all-target release clippy passed. The
  415-line probe was then removed.

The pre-gate correctly left 100k Fibonacci untouched and admitted only 72 of 100k uniform queries;
no strict-margin uniform proposal survived. Mega, bimodal, gradient, and outlier controls likewise
had no meaningful strict-margin saving. Clustered overshoot was real, but not scale-stable:

| Workload | Margin | Feasible sample vectors | Accepted | Center keys scanned | Net keys avoided | New tails / rescan dots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k clustered, seed 1 | 4x | 11,947 | 1,338 | 110,641 | 540,823 | 150 / 108,678 |
| 100k clustered, seed 2 | 4x | 12,151 | 780 | 48,793 | 240,441 | 125 / 107,228 |
| 100k clustered, seed 3 | 4x | 11,798 | 1,073 | 75,016 | 406,938 | 123 / 83,936 |
| 500k clustered, seed 1 | 4x | 57,208 | 1,227 | 63,442 | 179,853 | 218 / 182,768 |
| 100k clustered, seed 1 | 8x | 4,844 | 311 | 16,723 | 152,741 | 29 / 22,289 |
| 500k clustered, seed 1 | 8x | 24,950 | 58 | 2,222 | 9,158 | 14 / 14,500 |
| 100k splittable, seed 1 | 4x | 14,918 | 122 | 5,372 | 5,884 | 42 / 41,344 |
| 100k mega, default seed | 4x | 4,506 | 0 | 0 | 0 | 0 / 0 |

“Sample vectors” counts one SIMD dot operation per feasible query, not eight scalar dots. Even with
that favorable interpretation, the 500k clustered 4x form has only about 179.9k permanently
avoided keys against 57.2k sample-vector evaluations, 63.4k center-key visits, and at least 22.8k
eight-wide rescan chunks. That is roughly a 1.25x primitive-work margin before threshold selection,
partitioning, estimator control flow, and imperfect SIMD tails—not the requested 4x margin. At 8x,
the same scale samples 24,950 queries to accept 58 and avoids only 9,158 keys. Splittable is already
negative before selection overhead.

The narrow 100k clustered signal does not justify a scale-sensitive behavioral path. This closes
the high-threshold correction family for the current count model and lazy-tail representation. Do
not reopen it with another sampling geometry; a successor needs an already-available distribution
statistic or a representation in which changing the split does not require extra selection and
tail reconstruction.

## Seed-first packed-preparation oracle branch

- Branch: `agent/kernel-seed-first-oracle`
- Experiment: timing-only oracle after forwarded edge-check constraints and before stream
  consumption. Production still consumed the cached frontier normally.
- Exact gate: ask the existing termination certificate whether the already-prepared initial packed
  frontier proves that zero packed candidates are needed. Count the ordinary non-band dot row and
  high keys that a hypothetical micro-batched preparation could have omitted.
- Cheap gate: replace the prepared frontier bound with the existing conservative caps of every
  eligible grid cell plus the outside-neighborhood security bound. Validate the cap against the
  first retained candidate dot.
- Validation while the probe was present: timing-feature library tests passed (270 passed, 5
  ignored), timing-feature release clippy passed, and all reported cap-bound violations were zero.
  The 518-line probe was then removed.

The exact prepared-frontier ceiling is real but small. “Post-seed candidates” is work production
actually performed after a successful oracle result:

| Workload | Eligible rows | Exact hits | Hit row dots | Hit row keys | Post-seed candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100k Fibonacci | 100,000 | 4,399 | 453,702 / 11,615,720 (3.91%) | 5,761 / 1,287,986 (0.45%) | 2,529 |
| 100k uniform, seed 1 | 100,000 | 5,655 | 598,194 / 11,667,968 (5.13%) | 26,634 / 1,353,556 (1.97%) | 4,231 |
| 100k clustered, seed 1 | 94,563 | 4,902 | 1,269,011 / 42,482,658 (2.99%) | 234,669 / 16,346,361 (1.44%) | 4,610 |
| 100k splittable, seed 1 | 92,070 | 5,009 | 1,408,963 / 35,064,474 (4.02%) | 118,610 / 4,925,098 (2.41%) | 4,546 |
| 100k gradient, seed 1 | 100,000 | 6,313 | 2,989,816 / 47,719,773 (6.27%) | 19,644 / 1,341,017 (1.46%) | 4,430 |
| 100k mega, default seed | 100,000 | 103 | 32,681 / 89,563,732 (0.04%) | 407 / 1,286,994 (0.03%) | 68 |
| 500k clustered, seed 1 | 485,739 | 23,665 | 6,472,117 / 193,668,163 (3.34%) | 496,429 / 37,544,161 (1.32%) | 20,582 |

Total oracle hits and post-seed candidate counts differ because some oracle hits see an empty first
packed stage represented only by an upper bound; production already terminates those without
visiting a candidate. Across every row above, production visited exactly one later packed candidate
for each exact-batch hit. Therefore a simple pre-batch certificate, without restructuring
preparation, has only a one-candidate-per-exact-hit ceiling.

The existing cell caps are cheap enough to evaluate before row preparation, but the query's own
center-cell cap contains the generator and consequently has an upper bound near one. Useful hits
therefore concentrate on the final directed query in each grid cell, whose center suffix is empty:

| Workload | Cell-cap hits | Hit row dots | Hit row keys | Bound violations |
| --- | ---: | ---: | ---: | ---: |
| 100k Fibonacci | 1,635 | 166,289 (1.43%) | 133 (0.01%) | 0 |
| 100k uniform, seed 1 | 561 | 58,799 (0.50%) | 217 (0.02%) | 0 |
| 100k clustered, seed 1 | 149 | 25,212 (0.06%) | 1,031 (0.01%) | 0 |
| 100k splittable, seed 1 | 276 | 36,144 (0.10%) | 696 (0.01%) | 0 |
| 100k gradient, seed 1 | 295 | 109,485 (0.23%) | 78 (0.01%) | 0 |
| 100k mega, default seed | 112 | 104,918 (0.12%) | 6 (<0.01%) | 0 |
| 500k clustered, seed 1 | 1,271 | 231,105 (0.12%) | 4,058 (0.01%) | 0 |

Cell-cap hits can exceed prepared-frontier hits because the direct cap can be tighter than the
count-model threshold used as the packed frontier's conservative unseen bound. Zero first-dot
violations confirm that this is not an under-bound in the measured matrix.

The seed-first micro-batching hypothesis is closed for existing metadata. A finer precomputed
center-suffix decomposition could approach the 3--6% exact ceiling, but that ceiling is too small
to justify new cap storage, more preparation boundaries, and reduced group-wide SIMD without a
separate motivating workload. Do not build the behavioral form from hit counts alone.

## Same-cell regional local-hull oracle branch

- Branch: `agent/kernel-regional-hull-oracle`
- Experiment: timing-only census over each existing same-grid-cell generator group. For every
  group it recorded the exact union of distinct neighbor slots attempted by its cells, the sum of
  those attempts, and the union after adding the group's generators.
- Gate: compare repeated baseline candidate attempts with `p * (p - 1) / 2`, where `p` is the
  number of points the smallest same-cell local hull would contain. This is deliberately favorable
  to the proposal: it is only an arithmetic proxy for the current naive `O(p²)` hull, not a charge
  for its robust orientation predicates, face/horizon maintenance, a guard ring, certification,
  cell extraction, or stitching.
- Validation while present: the timing-enabled release benchmark built and completed the full
  workload matrix. The timing-only probe was then removed.

No group in the matrix reached the optimistic floor:

| Workload | Attempts | Duplicated attempts | Best group (rows / attempts / points) | Best attempts / pair floor | Qualifying groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100k Fibonacci | 721,993 | 515,913 | 17 / 121 / 35 | 20.34% | 0 |
| 100k uniform, seed 1 | 990,804 | 747,686 | 25 / 227 / 42 | 26.36% | 0 |
| 100k clustered, seed 1 | 5,824,287 | 5,207,306 | 1,339 / 739,404 / 1,490 | 66.65% | 0 |
| 100k splittable, seed 1 | 4,377,803 | 4,103,080 | 1,102 / 467,600 / 1,173 | 68.03% | 0 |
| 100k gradient, seed 1 | 992,070 | 791,962 | 2 / 13 / 8 | 46.43% | 0 |
| 100k mega, default seed | 4,371,527 | 2,988,912 | 2 / 9 / 6 | 60.00% | 0 |
| 100k great-circle, default jitter | 86,201,995 | 68,194,785 | 528 / 511,773 / 77,220 | 0.017% | 0 |
| 500k clustered, seed 1 | 16,993,127 | 14,593,205 | 1,037 / 481,454 / 1,138 | 74.42% | 0 |

The aggregate duplication is real, but it is attached to candidate unions large enough that the
current local hull's quadratic construction dominates. The `mega` maximum-work group is especially
diagnostic: it contains one row with 98,444 attempts and therefore offers no regional
amortization; a hull over its 98,445-point union would be pathological. Great-circle has broad
repetition, but its candidate unions are so large that even the optimistic floor misses by orders
of magnitude.

The same-grid-cell local-hull form is closed before guard-region replay. Enlarging regions is not
proved impossible by this census, but overlap alone is no longer a sufficient premise: a future
regional proposal needs a subquadratic construction or reusable triangulation, plus a cheap way to
form a much tighter certified candidate set. Do not add guard rings, replay machinery, or stitching
for the current `LocalHull` on the strength of duplicated-attempt counts.

## Group-shared shell traversal census

- Branch: `agent/kernel-shared-shell-oracle`
- Experiment: timing-only traces of the real shell frontier, aggregated at the existing
  same-grid-cell generator-group boundary after each query completed. The trace recorded every
  grid cell scanned, its BFS layer, and the number of resident slots whose dot was evaluated.
- Exact low-risk ceiling: compare per-query cell visits with the union of visited cells within the
  group. This is the repeated BFS visitation/neighbor-enumeration work that a shared layer schedule
  could remove while cells still clip and forward edge checks sequentially.
- Optimistic high ceiling: compare resident/query slot pairs with the sum of the maximum emitted
  residents for each union cell. This estimates position-data reuse for a multi-query kernel; it
  does **not** call the query-specific dot arithmetic, bounds, sorting, clipping, or dependency
  restructuring free.
- Divergence accounting: common prefix, per-group minimum/maximum trace and layer counts, active
  width thresholds, and groups whose traces were not simple prefixes were retained. A production
  form therefore cannot assume one lockstep cursor.
- Validation while present: `cargo test --release --features timing --lib` passed (270 passed, 5
  ignored). The probe was then removed; it never changed candidate order or construction behavior.

The structural overlap is large and survives scale:

| Workload | Multi-shell groups | Shell queries | Cell visits / union | Cell reuse ceiling | Resident/query slots / optimistic union | Position-load ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k Fibonacci | 0 | 0 | 0 / 0 | n/a | 0 / 0 | n/a |
| 100k uniform, seed 1 | 4,054 | 99,947 | 2,482,397 / 102,487 | 95.87% | 35,568,880 / 1,468,769 | 95.87% |
| 100k clustered, seed 1 | 2,956 | 99,038 | 2,521,759 / 91,008 | 96.39% | 77,335,225 / 1,271,065 | 98.36% |
| 100k mega, default seed | 862 | 81,065 | 11,482,468 / 744,280 | 93.52% | 478,947,438 / 2,438,920 | 99.49% |
| 100k great-circle, default jitter | 208 | 99,980 | 97,911,396 / 843,320 | 99.14% | 2,042,135,167 / 18,216,395 | 99.11% |
| 500k clustered, seed 1 | 15,815 | 496,812 | 12,693,912 / 480,409 | 96.22% | 342,651,200 / 6,585,338 | 98.08% |

At least 95% of the resident/query work in every positive row occurs in cells shared by 16 or more
active shell queries. The signal is not just a few pathological rows: 500k clustered retains
roughly 26 query visits per union cell and 52 resident/query pairs per optimistic unique resident.
Uniform is also diagnostic: only 357 shell batches reached clipping, but nearly every query entered
takeover far enough to scan a shell layer and obtain its bound. Existing `shell_layer_*` counters
therefore substantially understate the shell-frontier work available to this restructuring.

The gate promotes two deliberately separate implementations:

1. First test a shared group-local layer schedule while retaining sequential cell construction,
   query-specific bounds, exact candidate order, and all edge forwarding. Its ceiling is only the
   repeated BFS/stamp/neighbor-enumeration work, so accept it only on retired instructions and
   branches, not on the large resident ratio.
2. If the schedule result is modest, design a tiled resident-by-query dot/key kernel for groups
   with high active width. That can reuse point loads and expose SIMD across queries, but the dot
   count remains query-specific and current within-group forwarding creates a real dependency.
   The design must account for block boundaries, lost same-block seeds, extra speculative rows,
   per-query termination masks, sorting/storage traffic, and the broad great-circle divergence.

This is a promoted hypothesis, not an accepted optimization. Fibonacci is the natural zero-shell
control. Compare behavioral prototypes with single-threaded Linux `perf` counters on uniform,
clustered, mega, great-circle, and 500k clustered; defer cycle/wall conclusions while the machine
is noisy.
