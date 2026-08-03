use super::PhaseTimings;
use std::time::Duration;

impl PhaseTimings {
    pub(crate) fn report(&self, n: usize) {
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let total_ms = ms(self.total);

        let pct = |d: Duration| {
            if self.total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / self.total.as_secs_f64() * 100.0
            }
        };

        eprintln!("timing n={}", n);
        if self.preprocess.as_nanos() > 0 {
            eprintln!(
                "  preprocess:        {:7.1}ms ({:4.1}%)",
                ms(self.preprocess),
                pct(self.preprocess)
            );
        }
        eprintln!(
            "  knn_build:         {:7.1}ms ({:4.1}%)",
            ms(self.knn_build),
            pct(self.knn_build)
        );
        if let Some(sub) = &self.knn_build_sub {
            if sub.total().as_nanos() > 0 && self.knn_build.as_nanos() > 0 {
                let sub_pct = |d: Duration| d.as_secs_f64() / self.knn_build.as_secs_f64() * 100.0;
                eprintln!(
                    "    grid_count:      {:7.1}ms ({:4.1}%)",
                    ms(sub.count_cells),
                    sub_pct(sub.count_cells)
                );
                eprintln!(
                    "    grid_prefix:     {:7.1}ms ({:4.1}%)",
                    ms(sub.prefix_sum),
                    sub_pct(sub.prefix_sum)
                );
                eprintln!(
                    "    grid_scatter:    {:7.1}ms ({:4.1}%)",
                    ms(sub.scatter_soa),
                    sub_pct(sub.scatter_soa)
                );
                eprintln!(
                    "    grid_order_sample: pairs={} mean_abs_cell_delta={:.1} materialize_by_slot={}",
                    sub.input_order_pairs,
                    sub.input_order_abs_delta as f64 / sub.input_order_pairs.max(1) as f64,
                    sub.materialize_coordinates_by_slot,
                );
                eprintln!(
                    "    grid_neighbors:  {:7.1}ms ({:4.1}%)",
                    ms(sub.neighbors),
                    sub_pct(sub.neighbors)
                );
                eprintln!(
                    "    grid_ring2:      {:7.1}ms ({:4.1}%)",
                    ms(sub.ring2),
                    sub_pct(sub.ring2)
                );
                eprintln!(
                    "    grid_bounds:     {:7.1}ms ({:4.1}%)",
                    ms(sub.cell_bounds),
                    sub_pct(sub.cell_bounds)
                );
                if sub.topology_overlapped {
                    eprintln!(
                        "    grid_topology: overlapped (topology subphase elapsed times are non-additive)"
                    );
                } else if sub.topology_reused {
                    eprintln!("    grid_topology: reused from workspace");
                }
                eprintln!(
                    "    grid_security:   {:7.1}ms ({:4.1}%)",
                    ms(sub.security_3x3),
                    sub_pct(sub.security_3x3)
                );
            }
        }

        eprintln!(
            "  cell_construction: {:7.1}ms ({:4.1}%)",
            ms(self.cell_construction),
            pct(self.cell_construction)
        );

        // Estimate wall time contributions from per-cell CPU totals (parallel runs).
        let cpu_total = self.cell_sub.knn_query
            + self.cell_sub.packed_knn
            + self.cell_sub.clipping
            + self.cell_sub.certification
            + self.cell_sub.key_dedup
            + self.cell_sub.edge_collect
            + self.cell_sub.edge_resolve
            + self.cell_sub.edge_emit;
        let cpu_total_secs = cpu_total.as_secs_f64();
        let wall_secs = self.cell_construction.as_secs_f64();
        let cpu_to_wall = if cpu_total_secs > 0.0 {
            wall_secs / cpu_total_secs
        } else {
            1.0
        };
        let sub_pct = |d: Duration| {
            if cpu_total_secs > 0.0 {
                d.as_secs_f64() / cpu_total_secs * 100.0
            } else {
                0.0
            }
        };
        let est_wall_ms = |d: Duration| d.as_secs_f64() * cpu_to_wall * 1000.0;

        if cpu_total.as_nanos() > 0 {
            eprintln!(
                "    knn_query:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.knn_query),
                sub_pct(self.cell_sub.knn_query)
            );
            if self.cell_sub.packed_knn.as_nanos() > 0 {
                eprintln!(
                    "    packed_knn:      {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn),
                    sub_pct(self.cell_sub.packed_knn)
                );
                let kernel = [
                    ("setup", self.cell_sub.packed_setup),
                    ("security", self.cell_sub.packed_security),
                    ("center_pass", self.cell_sub.packed_center_pass),
                    ("ring_thresholds", self.cell_sub.packed_ring_thresholds),
                    ("ring_pass", self.cell_sub.packed_ring_pass),
                    ("ring_fallback", self.cell_sub.packed_ring_fallback),
                    ("select_prep", self.cell_sub.packed_select_prep),
                    ("select_partition", self.cell_sub.packed_select_partition),
                    ("select_sort", self.cell_sub.packed_select_sort),
                    ("select_scatter", self.cell_sub.packed_select_scatter),
                ];
                for (label, d) in kernel {
                    if d.as_nanos() > 0 {
                        eprintln!("      {:16} {:7.1}ms", label, est_wall_ms(d));
                    }
                }
                if self.cell_sub.packed_tail_builds > 0 {
                    eprintln!(
                        "      packed_builds: tail={}",
                        self.cell_sub.packed_tail_builds,
                    );
                    eprintln!(
                        "      tail_queries: possible={} requested={} ring_rescans={} empty={} ring_dot_evals={}",
                        self.cell_sub.tail_possible_queries,
                        self.cell_sub.tail_requested_queries,
                        self.cell_sub.ring_tail_rescans,
                        self.cell_sub.ring_tail_empty_rescans,
                        self.cell_sub.ring_tail_dot_evaluations,
                    );
                    eprintln!(
                        "      center_tail_candidates: total={} unrequested={} recomputed_dots={}",
                        self.cell_sub.center_tail_keys,
                        self.cell_sub.unused_center_tail_keys,
                        self.cell_sub.center_tail_dot_evaluations,
                    );
                    eprintln!(
                        "      chunk0_keys: total={} unused={}",
                        self.cell_sub.chunk0_keys, self.cell_sub.unused_chunk0_keys,
                    );
                }
                let packed_batch_classes =
                    ["chunk0_first", "chunk0_later", "tail_first", "tail_later"];
                for (class, label) in packed_batch_classes.into_iter().enumerate() {
                    let batches = self.cell_sub.packed_exact_batch_counts[class];
                    if batches > 0 {
                        eprintln!(
                            "      packed_batch_{label}: batches={} emitted={} visited={} abandoned={}",
                            batches,
                            self.cell_sub.packed_exact_slots_emitted[class],
                            self.cell_sub.packed_exact_slots_visited[class],
                            self.cell_sub.packed_exact_slots_abandoned[class],
                        );
                    }
                }
            }
            eprintln!(
                "    clipping:        {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.clipping),
                sub_pct(self.cell_sub.clipping)
            );
            eprintln!(
                "    certification:   {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.certification),
                sub_pct(self.cell_sub.certification)
            );
            eprintln!(
                "    key_dedup:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.key_dedup),
                sub_pct(self.cell_sub.key_dedup)
            );
            eprintln!(
                "    edge_collect:    {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.edge_collect),
                sub_pct(self.cell_sub.edge_collect)
            );
            eprintln!(
                "    edge_resolve:    {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.edge_resolve),
                sub_pct(self.cell_sub.edge_resolve)
            );
            eprintln!(
                "    edge_emit:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.edge_emit),
                sub_pct(self.cell_sub.edge_emit)
            );
            eprintln!(
                "    cells: used_knn={} knn_exhausted={} packed_tail_used={} packed_safe_exhausted={}",
                self.cell_sub.cells_used_knn,
                self.cell_sub.cells_knn_exhausted,
                self.cell_sub.cells_packed_tail_used,
                self.cell_sub.cells_packed_safe_exhausted
            );
            if self.cell_sub.shell_layer_batches > 0 {
                eprintln!(
                    "    shell_layers: batches={} slots={} prefix_consumed={} midlayer_terminations={}",
                    self.cell_sub.shell_layer_batches,
                    self.cell_sub.shell_layer_slots,
                    self.cell_sub.shell_layer_prefix_consumed,
                    self.cell_sub.shell_midlayer_terminations,
                );
            }
            if self.cell_sub.fallback_projection > 0
                || self.cell_sub.fallback_polygon_cap > 0
                || self.cell_sub.fallback_all_constraints > 0
            {
                eprintln!(
                    "    fallbacks: projection={} polygon_cap={} all_constraints={}",
                    self.cell_sub.fallback_projection,
                    self.cell_sub.fallback_polygon_cap,
                    self.cell_sub.fallback_all_constraints
                );
            }
            eprintln!(
                "    neighbors: mean={:.1} max={} (grid res={} max_occ={} rebuilt={})",
                self.cell_sub.neighbors_processed_total as f64 / n.max(1) as f64,
                self.cell_sub.neighbors_processed_max,
                self.grid_res,
                self.grid_max_occupancy,
                self.grid_rebuilt
            );
            let candidate_work = &self.cell_sub.candidate_work;
            if candidate_work.samples > 0 {
                eprintln!(
                    "    candidate_work: samples={} bucket_lb(p50={} p90={} p99={} p999={}) max={} relative_tail_lb(base={} 4x={} 16x={} 64x={})",
                    candidate_work.samples,
                    candidate_work.p50_bucket_lower,
                    candidate_work.p90_bucket_lower,
                    candidate_work.p99_bucket_lower,
                    candidate_work.p999_bucket_lower,
                    candidate_work.max,
                    candidate_work.relative_median_base,
                    candidate_work.count_ge_4x_median_lower,
                    candidate_work.count_ge_16x_median_lower,
                    candidate_work.count_ge_64x_median_lower,
                );
            }
            let no_progress_tail = &self.cell_sub.no_progress_tail;
            if no_progress_tail.samples > 0 || self.cell_sub.no_progress_tail_excluded > 0 {
                eprintln!(
                    "    no_progress_tail: samples={} excluded={} bucket_lb(p50={} p90={} p99={} p999={}) max={} relative_tail_lb(base={} 4x={} 16x={} 64x={})",
                    no_progress_tail.samples,
                    self.cell_sub.no_progress_tail_excluded,
                    no_progress_tail.p50_bucket_lower,
                    no_progress_tail.p90_bucket_lower,
                    no_progress_tail.p99_bucket_lower,
                    no_progress_tail.p999_bucket_lower,
                    no_progress_tail.max,
                    no_progress_tail.relative_median_base,
                    no_progress_tail.count_ge_4x_median_lower,
                    no_progress_tail.count_ge_16x_median_lower,
                    no_progress_tail.count_ge_64x_median_lower,
                );
            }
            let examine_per_edge = if self.cell_sub.final_edges_total > 0 {
                self.cell_sub.neighbors_processed_total as f64
                    / self.cell_sub.final_edges_total as f64
            } else {
                0.0
            };
            eprintln!(
                "    final_edges: mean={:.2} max={} examine_per_edge={:.3}",
                self.cell_sub.final_edges_total as f64 / n.max(1) as f64,
                self.cell_sub.final_edges_max,
                examine_per_edge
            );
            if self.cell_sub.directional_shadow_checks > 0 {
                eprintln!(
                    "    dir_shadow: checks={} tests={} hits={} saved={} support_tests={} support_hits={} support_saved={} support_false_pos={}",
                    self.cell_sub.directional_shadow_checks,
                    self.cell_sub.directional_shadow_candidate_tests,
                    self.cell_sub.directional_shadow_hits,
                    self.cell_sub.directional_shadow_saved,
                    self.cell_sub.directional_support_candidate_tests,
                    self.cell_sub.directional_support_hits,
                    self.cell_sub.directional_support_saved,
                    self.cell_sub.directional_support_false_positive_hits
                );
            }
        }

        eprintln!(
            "  dedup:             {:7.1}ms ({:4.1}%)",
            ms(self.dedup),
            pct(self.dedup)
        );
        eprintln!(
            "    keys: triplet={} edge_mismatches={} primary_refs={} foreign_overrides={} ({:.3}%)",
            self.dedup_sub.triplet_keys,
            self.dedup_sub.edge_mismatches_count,
            self.dedup_sub.primary_cell_references,
            self.dedup_sub.reference_overrides,
            100.0 * self.dedup_sub.reference_overrides as f64
                / self.dedup_sub.primary_cell_references.max(1) as f64,
        );
        eprintln!(
            "    assembly: bookkeeping={:.3}ms overflow={:.3}ms deferred={:.3}ms finalize={:.3}ms vertices={:.3}ms prefixes={:.3}ms incidence={:.3}ms indices={:.3}ms overrides={:.3}ms zero_hints={:.3}ms",
            ms(self.dedup_sub.bookkeeping),
            ms(self.dedup_sub.edge_check_overflow),
            ms(self.dedup_sub.deferred_patching),
            ms(self.dedup_sub.finalize_shards),
            ms(self.dedup_sub.concat_vertices),
            ms(self.dedup_sub.emit_cell_prefixes),
            ms(self.dedup_sub.incidence_summary),
            ms(self.dedup_sub.scatter_cell_indices),
            ms(self.dedup_sub.patch_reference_overrides),
            ms(self.dedup_sub.exact_zero_hints),
        );
        eprintln!(
            "    overflow: records={} sort={:.3}ms match={:.3}ms",
            self.dedup_sub.edge_check_overflow_records,
            ms(self.dedup_sub.edge_check_overflow_sort),
            ms(self.dedup_sub.edge_check_overflow_match),
        );
        eprintln!(
            "    shard_order_sample: descents={} / {} ({:.3}%) mean_abs_global_delta={:.1} scatter_by_shard={}",
            self.dedup_sub.shard_order_descents,
            self.dedup_sub.shard_order_pairs,
            100.0 * self.dedup_sub.shard_order_descents as f64
                / self.dedup_sub.shard_order_pairs.max(1) as f64,
            self.dedup_sub.shard_order_abs_delta as f64
                / self.dedup_sub.shard_order_pairs.max(1) as f64,
            self.dedup_sub.scatter_by_shard,
        );
        eprintln!(
            "  edge_reconcile:    {:7.1}ms ({:4.1}%)",
            ms(self.edge_reconcile),
            pct(self.edge_reconcile)
        );
        if self.merge_safety_scan_cells > 0 || self.merge_safety_global_fallbacks > 0 {
            eprintln!(
                "    merge_safety: scanned_cells={} global_fallbacks={}",
                self.merge_safety_scan_cells, self.merge_safety_global_fallbacks
            );
        }
        eprintln!(
            "  assemble:          {:7.1}ms ({:4.1}%)",
            ms(self.assemble),
            pct(self.assemble)
        );
        eprintln!(
            "  output_resolution: mode={} drift_fallback={} local_scan(reconcile_cells={},rebuild_cells={}) hint_cells={} hinted_candidates={} detected_edges={}",
            if self.resolution_drift_fallback {
                "exhaustive"
            } else {
                "certified_hint"
            },
            self.resolution_drift_fallback as u8,
            self.resolution_reconcile_scan_cells,
            self.resolution_rebuild_scan_cells,
            self.resolution_hint_cells,
            self.resolution_hinted_candidates,
            self.resolution_detected_edges,
        );

        if std::env::var_os("VORONOI_MESH_TIMING_KV").is_some() {
            let (grid_order_pairs, grid_order_abs_delta, grid_materialize_by_slot) = self
                .knn_build_sub
                .as_ref()
                .map(|sub| {
                    (
                        sub.input_order_pairs,
                        sub.input_order_abs_delta,
                        sub.materialize_coordinates_by_slot as u8,
                    )
                })
                .unwrap_or((0, 0, 0));
            eprintln!(
                "TIMING_KV n={n} total_ms={total:.3} preprocess_ms={pre:.3} weld_pairs={wp} weld_pair_capacity={wpc} knn_build_ms={kb:.3} grid_order_pairs={gop} grid_order_abs_delta={goa} grid_materialize_by_slot={gms} cell_construction_ms={cc:.3} dedup_ms={dd:.3} dedup_bookkeeping_ms={dbk:.3} dedup_overflow_ms={dof:.3} dedup_deferred_ms={ddp:.3} dedup_finalize_ms={dfs:.3} dedup_vertices_ms={dvt:.3} dedup_prefixes_ms={dcp:.3} dedup_incidence_ms={dis:.3} dedup_indices_ms={dci:.3} dedup_overrides_ms={dro:.3} dedup_zero_hints_ms={dzh:.3} shard_order_descents={sod} shard_order_pairs={sop} shard_order_abs_delta={soa} scatter_by_shard={sbs} edge_reconcile_ms={er:.3} merge_safety_scan_cells={mssc} merge_safety_global_fallbacks={msgf} assemble_ms={asmb:.3} resolution_certified_hint={rch} resolution_fallback_drift={rfd} resolution_reconcile_scan_cells={rrsc} resolution_rebuild_scan_cells={rpsc} resolution_hint_cells={rhc} resolution_hinted_candidates={rhcand} resolution_detected_edges={rde} cells_used_knn={cuk} cells_packed_tail_used={cpt} fallback_projection={fpj} fallback_polygon_cap={fpc} fallback_all_constraints={fac} packed_total_ms={ptm:.3} packed_select_partition_ms={psp:.3} packed_select_sort_ms={pss:.3} packed_select_scatter_ms={psc:.3} packed_tail_builds={ptb} packed_keys_materialized={pkm} packed_key_capacity_peak={pkp} tail_possible_queries={tpq} tail_requested_queries={trq} ring_tail_rescans={rtr} ring_tail_empty_rescans={rte} ring_tail_dot_evaluations={rtd} center_tail_keys={ctk} unused_center_tail_keys={uctk} center_tail_dot_evaluations={ctd} chunk0_keys={c0k} unused_chunk0_keys={uc0k} packed_chunk0_first_batches={pc0fb} packed_chunk0_first_emitted={pc0fe} packed_chunk0_first_visited={pc0fv} packed_chunk0_first_abandoned={pc0fa} packed_chunk0_later_batches={pc0lb} packed_chunk0_later_emitted={pc0le} packed_chunk0_later_visited={pc0lv} packed_chunk0_later_abandoned={pc0la} packed_tail_first_batches={ptfb} packed_tail_first_emitted={ptfe} packed_tail_first_visited={ptfv} packed_tail_first_abandoned={ptfa} packed_tail_later_batches={ptlb} packed_tail_later_emitted={ptle} packed_tail_later_visited={ptlv} packed_tail_later_abandoned={ptla} shell_layer_batches={slb} shell_layer_slots={sls} shell_layer_prefix_consumed={slp} shell_midlayer_terminations={slm} neighbors_total={nt} neighbors_max={nm} candidate_work_samples={cws} candidate_work_p50_lb={cw50} candidate_work_p90_lb={cw90} candidate_work_p99_lb={cw99} candidate_work_p999_lb={cw999} candidate_work_max={cwm} candidate_work_relative_base={cwb} candidate_work_ge4x_median_lb={cw4} candidate_work_ge16x_median_lb={cw16} candidate_work_ge64x_median_lb={cw64} no_progress_tail_samples={nps} no_progress_tail_excluded={npx} no_progress_tail_p50_lb={np50} no_progress_tail_p90_lb={np90} no_progress_tail_p99_lb={np99} no_progress_tail_p999_lb={np999} no_progress_tail_max={npm} no_progress_tail_relative_base={npb} no_progress_tail_ge4x_median_lb={np4} no_progress_tail_ge16x_median_lb={np16} no_progress_tail_ge64x_median_lb={np64} final_edges_total={fet} final_edges_max={fem} examine_per_edge={epe:.6} dir_shadow_checks={dsc} dir_shadow_candidate_tests={dst} dir_shadow_hits={dsh} dir_shadow_saved={dss} dir_support_candidate_tests={dpt} dir_support_hits={dph} dir_support_saved={dps} dir_support_false_positive_hits={dpf} grid_res={gr} grid_max_occ={gmo} grid_rebuilt={grb}",
                n = n,
                total = total_ms,
                pre = ms(self.preprocess),
                wp = self.weld_pairs,
                wpc = self.weld_pair_capacity,
                kb = ms(self.knn_build),
                gop = grid_order_pairs,
                goa = grid_order_abs_delta,
                gms = grid_materialize_by_slot,
                cc = ms(self.cell_construction),
                dd = ms(self.dedup),
                dbk = ms(self.dedup_sub.bookkeeping),
                dof = ms(self.dedup_sub.edge_check_overflow),
                ddp = ms(self.dedup_sub.deferred_patching),
                dfs = ms(self.dedup_sub.finalize_shards),
                dvt = ms(self.dedup_sub.concat_vertices),
                dcp = ms(self.dedup_sub.emit_cell_prefixes),
                dis = ms(self.dedup_sub.incidence_summary),
                dci = ms(self.dedup_sub.scatter_cell_indices),
                dro = ms(self.dedup_sub.patch_reference_overrides),
                dzh = ms(self.dedup_sub.exact_zero_hints),
                sod = self.dedup_sub.shard_order_descents,
                sop = self.dedup_sub.shard_order_pairs,
                soa = self.dedup_sub.shard_order_abs_delta,
                sbs = self.dedup_sub.scatter_by_shard as u8,
                er = ms(self.edge_reconcile),
                mssc = self.merge_safety_scan_cells,
                msgf = self.merge_safety_global_fallbacks,
                asmb = ms(self.assemble),
                rch = (!self.resolution_drift_fallback) as u8,
                rfd = self.resolution_drift_fallback as u8,
                rrsc = self.resolution_reconcile_scan_cells,
                rpsc = self.resolution_rebuild_scan_cells,
                rhc = self.resolution_hint_cells,
                rhcand = self.resolution_hinted_candidates,
                rde = self.resolution_detected_edges,
                cuk = self.cell_sub.cells_used_knn,
                cpt = self.cell_sub.cells_packed_tail_used,
                fpj = self.cell_sub.fallback_projection,
                fpc = self.cell_sub.fallback_polygon_cap,
                fac = self.cell_sub.fallback_all_constraints,
                ptm = ms(self.cell_sub.packed_knn),
                psp = ms(self.cell_sub.packed_select_partition),
                pss = ms(self.cell_sub.packed_select_sort),
                psc = ms(self.cell_sub.packed_select_scatter),
                ptb = self.cell_sub.packed_tail_builds,
                pkm = self.cell_sub.packed_keys_materialized,
                pkp = self.cell_sub.packed_key_capacity_peak,
                tpq = self.cell_sub.tail_possible_queries,
                trq = self.cell_sub.tail_requested_queries,
                rtr = self.cell_sub.ring_tail_rescans,
                rte = self.cell_sub.ring_tail_empty_rescans,
                rtd = self.cell_sub.ring_tail_dot_evaluations,
                ctk = self.cell_sub.center_tail_keys,
                uctk = self.cell_sub.unused_center_tail_keys,
                ctd = self.cell_sub.center_tail_dot_evaluations,
                c0k = self.cell_sub.chunk0_keys,
                uc0k = self.cell_sub.unused_chunk0_keys,
                pc0fb = self.cell_sub.packed_exact_batch_counts[0],
                pc0fe = self.cell_sub.packed_exact_slots_emitted[0],
                pc0fv = self.cell_sub.packed_exact_slots_visited[0],
                pc0fa = self.cell_sub.packed_exact_slots_abandoned[0],
                pc0lb = self.cell_sub.packed_exact_batch_counts[1],
                pc0le = self.cell_sub.packed_exact_slots_emitted[1],
                pc0lv = self.cell_sub.packed_exact_slots_visited[1],
                pc0la = self.cell_sub.packed_exact_slots_abandoned[1],
                ptfb = self.cell_sub.packed_exact_batch_counts[2],
                ptfe = self.cell_sub.packed_exact_slots_emitted[2],
                ptfv = self.cell_sub.packed_exact_slots_visited[2],
                ptfa = self.cell_sub.packed_exact_slots_abandoned[2],
                ptlb = self.cell_sub.packed_exact_batch_counts[3],
                ptle = self.cell_sub.packed_exact_slots_emitted[3],
                ptlv = self.cell_sub.packed_exact_slots_visited[3],
                ptla = self.cell_sub.packed_exact_slots_abandoned[3],
                slb = self.cell_sub.shell_layer_batches,
                sls = self.cell_sub.shell_layer_slots,
                slp = self.cell_sub.shell_layer_prefix_consumed,
                slm = self.cell_sub.shell_midlayer_terminations,
                nt = self.cell_sub.neighbors_processed_total,
                nm = self.cell_sub.neighbors_processed_max,
                cws = self.cell_sub.candidate_work.samples,
                cw50 = self.cell_sub.candidate_work.p50_bucket_lower,
                cw90 = self.cell_sub.candidate_work.p90_bucket_lower,
                cw99 = self.cell_sub.candidate_work.p99_bucket_lower,
                cw999 = self.cell_sub.candidate_work.p999_bucket_lower,
                cwm = self.cell_sub.candidate_work.max,
                cwb = self.cell_sub.candidate_work.relative_median_base,
                cw4 = self.cell_sub.candidate_work.count_ge_4x_median_lower,
                cw16 = self.cell_sub.candidate_work.count_ge_16x_median_lower,
                cw64 = self.cell_sub.candidate_work.count_ge_64x_median_lower,
                nps = self.cell_sub.no_progress_tail.samples,
                npx = self.cell_sub.no_progress_tail_excluded,
                np50 = self.cell_sub.no_progress_tail.p50_bucket_lower,
                np90 = self.cell_sub.no_progress_tail.p90_bucket_lower,
                np99 = self.cell_sub.no_progress_tail.p99_bucket_lower,
                np999 = self.cell_sub.no_progress_tail.p999_bucket_lower,
                npm = self.cell_sub.no_progress_tail.max,
                npb = self.cell_sub.no_progress_tail.relative_median_base,
                np4 = self.cell_sub.no_progress_tail.count_ge_4x_median_lower,
                np16 = self.cell_sub.no_progress_tail.count_ge_16x_median_lower,
                np64 = self.cell_sub.no_progress_tail.count_ge_64x_median_lower,
                fet = self.cell_sub.final_edges_total,
                fem = self.cell_sub.final_edges_max,
                epe = if self.cell_sub.final_edges_total > 0 {
                    self.cell_sub.neighbors_processed_total as f64
                        / self.cell_sub.final_edges_total as f64
                } else {
                    0.0
                },
                dsc = self.cell_sub.directional_shadow_checks,
                dst = self.cell_sub.directional_shadow_candidate_tests,
                dsh = self.cell_sub.directional_shadow_hits,
                dss = self.cell_sub.directional_shadow_saved,
                dpt = self.cell_sub.directional_support_candidate_tests,
                dph = self.cell_sub.directional_support_hits,
                dps = self.cell_sub.directional_support_saved,
                dpf = self.cell_sub.directional_support_false_positive_hits,
                gr = self.grid_res,
                gmo = self.grid_max_occupancy,
                grb = self.grid_rebuilt as u8,
            );
        }
    }
}
