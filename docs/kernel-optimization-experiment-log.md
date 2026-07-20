# Kernel optimization experiment log

This log records the read-only review synthesis, measurement gates, and branch outcomes from the
July 2026 kernel pass. Wall-clock measurements taken while the shared machine is busy are
non-decisive. Retired instructions and branches are the primary behavioral signals; timing-only
counts are used for workload censuses.

## Pass closeout

The multi-model shortlist is exhausted. The measurement infrastructure and read-only oracles were
retained; every production behavior experiment was rejected and reverted. The branch names below
are archival experiment labels, not outstanding merge candidates.

| Family | Branches | Final decision |
| --- | --- | --- |
| Fixed smaller packed prefix | `agent/kernel-demand-prefix` | Rejected: the 8-slot first batch saved 0.315% instructions on Fibonacci but added 2.777% on uniform and increased branches on both. |
| Compact high-key overflow | `agent/kernel-compact-overflow`, `agent/kernel-compact-top64-cost`, `agent/kernel-compact-overflow-rebuild` | Shadow census retained; heap-based additive and true-replacement forms rejected at +5.1--11.4% and +6.9--7.7% instructions respectively. |
| Shell-cell rejection | `agent/kernel-shell-cell-reject`, `agent/kernel-shell-cell-cap`, `agent/kernel-shell-cell-cap-skip` | Exact and conservative-cap oracles retained; the order-preserving production form was rejected at +1.8--3.5% instructions. |

The negative results share a useful conclusion: the existing width-one unchanged clip and
append-then-partition selection paths are already cheap. A successor should remove dot/key work
before those paths, change the cutoff without per-key maintenance, or replace repeated per-cell
work with a regional algorithm. The untried follow-ups are recorded in
[`algorithmic-performance-ideas.md`](algorithmic-performance-ideas.md#post-review-kernel-hypotheses-untried);
`PERF-002` in [`work-log.md`](work-log.md#perf-002--post-review-kernel-hypotheses) owns the next
measurement gate.

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
