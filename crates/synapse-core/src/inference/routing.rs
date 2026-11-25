//! Bounded private topology selection. Latency predictions use only matching
//! real decode observations. Cold routes expose no invented millisecond cost.
use crate::shard::{BoundedShardParticipant, plan_bounded_shards};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{
    collections::{BTreeSet, HashSet},
    ops::Range,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeObservation {
    pub layers: usize,
    pub context_limit: usize,
    /// Local IO/compute, or complete authenticated remote request/response.
    pub elapsed_micros: u64,
    pub sampled_at_unix: u64,
}
#[derive(Clone, Debug)]
pub struct RouteCandidate {
    pub id: String,
    pub max_layers: usize,
    pub measured_compute_score: u64,
    pub measured_rtt_micros: Option<u64>,
    pub observed_decode: Option<DecodeObservation>,
}
#[derive(Clone, Debug, Serialize)]
pub struct SelectedStage {
    pub id: String,
    pub layer_range: Range<usize>,
}
#[derive(Clone, Debug, Serialize)]
pub struct RouteDecision {
    pub mode: String,
    pub considered_routes: usize,
    pub selected_remote_stages: usize,
    pub predicted_decode_micros: Option<u64>,
    pub measured_remote_links: usize,
    pub unknown_remote_links: usize,
    pub explanation: String,
}
#[derive(Clone, Debug)]
pub struct SelectedRoute {
    pub stages: Vec<SelectedStage>,
    pub decision: RouteDecision,
}
struct Evaluated {
    stages: Vec<SelectedStage>,
    remote_stages: usize,
    predicted: Option<u64>,
    unknown_links: usize,
    rtt_sum: u64,
    compute_proxy: u128,
    signature: String,
}
impl Evaluated {
    fn rank(&self) -> (bool, u64, usize, usize, u64, u128, &str) {
        (
            self.predicted.is_none(),
            self.predicted.unwrap_or(0),
            self.remote_stages,
            self.unknown_links,
            self.rtt_sum,
            self.compute_proxy,
            &self.signature,
        )
    }
}
/// Examine at most 256 worker subsets and 1024 layers. Local and one-worker
/// whole-transformer routes are included; no unused peer is forced into a plan.
/// Up to eight approved candidates are supported; larger pools require caller
/// paging/selection, rather than unbounded combinatorial search.
pub fn select_route(
    total_layers: usize,
    coordinator: &RouteCandidate,
    workers: &[RouteCandidate],
    max_remote_stages: usize,
    context_limit: usize,
    now: u64,
) -> Result<SelectedRoute> {
    ensure!(
        (1..=1024).contains(&total_layers)
            && workers.len() <= 8
            && (1..=8).contains(&max_remote_stages),
        "route search bounds exceeded"
    );
    ensure!(
        context_limit > 0 && coordinator.measured_compute_score > 0,
        "missing context or measured coordinator capability"
    );
    let mut ids = HashSet::from([coordinator.id.clone()]);
    for worker in workers {
        ensure!(ids.insert(worker.id.clone()), "duplicate route identity");
    }
    let mut seen = BTreeSet::new();
    let mut feasible = Vec::new();
    for subset in 0usize..(1usize << workers.len()) {
        if subset.count_ones() as usize > max_remote_stages {
            continue;
        }
        let selected: Vec<_> = std::iter::once(coordinator)
            .chain(
                workers
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| subset & (1 << i) != 0)
                    .map(|(_, c)| c),
            )
            .collect();
        let participants: Vec<_> = selected
            .iter()
            .map(|c| BoundedShardParticipant {
                participant: c.id.clone(),
                compute_score: c.measured_compute_score,
                max_layers: c.max_layers,
            })
            .collect();
        let Ok(assignments) = plan_bounded_shards(total_layers, &participants) else {
            continue;
        };
        let stages: Vec<_> = assignments
            .into_iter()
            .map(|a| SelectedStage {
                id: a.participant,
                layer_range: a.layer_range,
            })
            .collect();
        let signature = stages
            .iter()
            .map(|s| format!("{}:{}-{}", s.id, s.layer_range.start, s.layer_range.end))
            .collect::<Vec<_>>()
            .join(";");
        if !seen.insert(signature.clone()) {
            continue;
        }
        let mut remote_stages = 0;
        let mut unknown_links = 0;
        let mut rtt_sum = 0u64;
        let mut compute_proxy = 0u128;
        // Coordinator IO remains mandatory even when it owns no layers.
        let mut predicted = Some(0u64);
        for candidate in &selected {
            let layers = stages
                .iter()
                .find(|s| s.id == candidate.id)
                .map_or(0, |s| s.layer_range.len());
            if candidate.id != coordinator.id && layers == 0 {
                continue;
            }
            if candidate.id != coordinator.id {
                remote_stages += 1;
                if let Some(rtt) = candidate.measured_rtt_micros {
                    rtt_sum = rtt_sum.saturating_add(rtt);
                } else {
                    unknown_links += 1;
                }
            }
            compute_proxy = compute_proxy.saturating_add(
                (layers as u128) * 1_000_000_000u128
                    / (candidate.measured_compute_score.max(1) as u128),
            );
            let matching = candidate.observed_decode.as_ref().filter(|sample| {
                sample.layers == layers
                    && sample.context_limit == context_limit
                    && sample.elapsed_micros > 0
                    && sample.sampled_at_unix <= now
                    && now - sample.sampled_at_unix <= 300
            });
            predicted = predicted
                .zip(matching)
                .and_then(|(sum, sample)| sum.checked_add(sample.elapsed_micros));
        }
        feasible.push(Evaluated {
            stages,
            remote_stages,
            predicted,
            unknown_links,
            rtt_sum,
            compute_proxy,
            signature,
        });
    }
    let considered_routes = feasible.len();
    let best = feasible
        .into_iter()
        .min_by(|a, b| a.rank().cmp(&b.rank()))
        .ok_or_else(|| anyhow::anyhow!("no memory-feasible route within the stage limit"))?;
    let mode = if best.remote_stages == 0 {
        "local"
    } else if best.remote_stages == 1 && best.stages.iter().all(|s| s.id != coordinator.id) {
        "one_worker"
    } else {
        "contiguous_shards"
    }
    .to_owned();
    let explanation=if best.predicted.is_some() {
        "Selected using recent measured decode durations for these exact layer counts and context limit; remote durations include transfer and compute. Predictions are workload estimates, not WAN bandwidth measurements."
    } else {
        "No complete matching decode calibration exists. Selected a feasible route with few remote stages, measured RTT where available, then measured compute probes. Millisecond latency and bandwidth remain unknown."
    }.to_owned();
    Ok(SelectedRoute {
        stages: best.stages,
        decision: RouteDecision {
            mode,
            considered_routes,
            selected_remote_stages: best.remote_stages,
            predicted_decode_micros: best.predicted,
            measured_remote_links: best.remote_stages - best.unknown_links,
            unknown_remote_links: best.unknown_links,
            explanation,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(id: &str, layers: usize, score: u64, rtt: Option<u64>) -> RouteCandidate {
        RouteCandidate {
            id: id.into(),
            max_layers: layers,
            measured_compute_score: score,
            measured_rtt_micros: rtt,
            observed_decode: None,
        }
    }
    #[test]
    fn complete_fast_worker_excludes_unnecessary_slow_stage() {
        let local = c("local", 0, 100, None);
        let workers = [
            c("fast", 8, 1000, Some(1000)),
            c("slow", 8, 10, Some(90_000)),
        ];
        let route = select_route(8, &local, &workers, 4, 32, 100).unwrap();
        assert_eq!(route.stages.len(), 1);
        assert_eq!(route.stages[0].id, "fast");
        assert!(route.decision.predicted_decode_micros.is_none());
        assert_eq!(route.decision.mode, "one_worker");
    }
    #[test]
    fn heterogeneous_memory_can_require_two_workers_and_stage_limit_is_hard() {
        let local = c("local", 0, 100, None);
        let workers = [c("a", 3, 100, Some(3000)), c("b", 5, 200, Some(8000))];
        assert!(select_route(8, &local, &workers, 1, 32, 100).is_err());
        let route = select_route(8, &local, &workers, 2, 32, 100).unwrap();
        assert_eq!(route.stages[0].layer_range, 0..3);
        assert_eq!(route.stages[1].layer_range, 3..8);
    }
    #[test]
    fn local_and_whole_worker_are_candidates_and_unknown_is_not_fake_zero_rtt() {
        let local = c("local", 8, 100, None);
        let workers = [c("unknown", 8, 1000, None), c("known", 8, 100, Some(10))];
        assert_eq!(
            select_route(8, &local, &workers, 2, 32, 100)
                .unwrap()
                .decision
                .mode,
            "local"
        );
        let mut coordinator = local;
        coordinator.max_layers = 0;
        assert_eq!(
            select_route(8, &coordinator, &workers, 2, 32, 100)
                .unwrap()
                .stages[0]
                .id,
            "known"
        );
    }
    #[test]
    fn measured_matching_decode_changes_choice_and_stale_measurement_cannot() {
        let mut local = c("local", 0, 100, None);
        local.observed_decode = Some(DecodeObservation {
            layers: 0,
            context_limit: 32,
            elapsed_micros: 50,
            sampled_at_unix: 100,
        });
        let mut a = c("a", 8, 100, Some(10));
        let mut b = c("b", 8, 100, Some(100));
        a.observed_decode = Some(DecodeObservation {
            layers: 8,
            context_limit: 32,
            elapsed_micros: 10000,
            sampled_at_unix: 100,
        });
        b.observed_decode = Some(DecodeObservation {
            layers: 8,
            context_limit: 32,
            elapsed_micros: 1000,
            sampled_at_unix: 100,
        });
        let route = select_route(8, &local, &[a.clone(), b.clone()], 2, 32, 100).unwrap();
        assert_eq!(route.stages[0].id, "b");
        assert_eq!(route.decision.predicted_decode_micros, Some(1050));
        let route = select_route(8, &local, &[a, b], 2, 32, 401).unwrap();
        assert_eq!(route.stages[0].id, "a");
        assert!(route.decision.predicted_decode_micros.is_none());
    }
    #[test]
    fn deterministic_ties_and_duplicate_id_rejection() {
        let local = c("local", 0, 100, None);
        let a = c("a", 8, 100, Some(10));
        let b = c("b", 8, 100, Some(10));
        assert_eq!(
            select_route(8, &local, &[a.clone(), b.clone()], 2, 32, 100)
                .unwrap()
                .stages[0]
                .id,
            "a"
        );
        assert_eq!(
            select_route(8, &local, &[b, a.clone()], 2, 32, 100)
                .unwrap()
                .stages[0]
                .id,
            "a"
        );
        assert!(select_route(8, &local, &[a.clone(), a], 2, 32, 100).is_err());
    }
}
