use std::ops::Range;

use crate::{DType, ShardPlan, protocol::ProtocolError};
use serde::{Deserialize, Serialize};

pub const RAM_PER_LAYER_HEURISTIC_MB: u64 = 50;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn new(start: u32, end: u32) -> Result<Self, ProtocolError> {
        if start >= end {
            return Err(ProtocolError::InvalidShardPlan {
                reason: "layer range must be non-empty",
            });
        }

        Ok(Self { start, end })
    }

    pub fn len(self) -> u32 {
        self.end - self.start
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticShardAssignment<P> {
    pub participant: P,
    pub layer_range: Range<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeCapability {
    pub compute_score: u64,
    pub available_ram_mb: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightedShardParticipant<P> {
    pub participant: P,
    pub capability: NodeCapability,
}

pub fn plan_static_shards<P: Clone>(
    total_layers: usize,
    participants: &[P],
) -> Result<Vec<StaticShardAssignment<P>>, ProtocolError> {
    if total_layers == 0 {
        return Err(ProtocolError::InvalidShardPlan {
            reason: "total layer count must be greater than zero",
        });
    }

    if participants.is_empty() {
        return Err(ProtocolError::InvalidShardPlan {
            reason: "at least one participant is required",
        });
    }

    let base_layers = total_layers / participants.len();
    let remainder = total_layers % participants.len();
    let mut cursor = 0usize;
    let mut assignments = Vec::with_capacity(participants.len().min(total_layers));

    for (index, participant) in participants.iter().enumerate() {
        let shard_len = base_layers + usize::from(index < remainder);
        let start = cursor;
        let end = start + shard_len;
        cursor = end;

        if shard_len == 0 {
            continue;
        }

        assignments.push(StaticShardAssignment {
            participant: participant.clone(),
            layer_range: start..end,
        });
    }

    debug_assert_eq!(cursor, total_layers);
    Ok(assignments)
}

#[derive(Clone, Debug)]
struct WeightedNodeState<P> {
    participant: P,
    compute_score: u64,
    max_layers: usize,
    assigned_layers: usize,
    ideal_numerator: u128,
}

impl<P> WeightedNodeState<P> {
    fn deviation_numerator(&self, total_compute: i128) -> i128 {
        self.ideal_numerator as i128 - (self.assigned_layers as i128 * total_compute)
    }
}

pub fn plan_weighted_shards<P: Clone>(
    total_layers: usize,
    participants: &[WeightedShardParticipant<P>],
) -> Result<Vec<StaticShardAssignment<P>>, ProtocolError> {
    if total_layers == 0 {
        return Err(ProtocolError::InvalidShardPlan {
            reason: "total layer count must be greater than zero",
        });
    }

    if participants.is_empty() {
        return Err(ProtocolError::InvalidShardPlan {
            reason: "at least one participant is required",
        });
    }

    let total_compute = participants
        .iter()
        .map(|participant| participant.capability.compute_score as u128)
        .sum::<u128>();
    if total_compute == 0 {
        return Err(ProtocolError::InvalidShardPlan {
            reason: "weighted planner requires non-zero total compute",
        });
    }

    let mut states = participants
        .iter()
        .map(|participant| {
            let ideal_numerator =
                (total_layers as u128).saturating_mul(participant.capability.compute_score as u128);
            let rounded_layers = (ideal_numerator + (total_compute / 2)) / total_compute;
            let max_layers = (participant.capability.available_ram_mb / RAM_PER_LAYER_HEURISTIC_MB)
                .try_into()
                .unwrap_or(usize::MAX);

            WeightedNodeState {
                participant: participant.participant.clone(),
                compute_score: participant.capability.compute_score,
                max_layers,
                assigned_layers: rounded_layers
                    .try_into()
                    .unwrap_or(total_layers)
                    .min(max_layers),
                ideal_numerator,
            }
        })
        .collect::<Vec<_>>();

    let total_capacity = states
        .iter()
        .map(|state| state.max_layers)
        .fold(0usize, usize::saturating_add);
    if total_capacity < total_layers {
        return Err(ProtocolError::InvalidShardPlan {
            reason: "insufficient RAM capacity for requested layer count",
        });
    }

    let total_compute_i128 = total_compute as i128;
    let mut assigned_total = states
        .iter()
        .map(|state| state.assigned_layers)
        .fold(0usize, usize::saturating_add);

    while assigned_total > total_layers {
        let mut selected_index: Option<usize> = None;
        for (index, state) in states.iter().enumerate() {
            if state.assigned_layers == 0 {
                continue;
            }

            let state_deviation = state.deviation_numerator(total_compute_i128);
            let is_better = match selected_index {
                None => true,
                Some(best_index) => {
                    let best = &states[best_index];
                    let best_deviation = best.deviation_numerator(total_compute_i128);
                    state_deviation < best_deviation
                        || (state_deviation == best_deviation
                            && (state.compute_score < best.compute_score
                                || (state.compute_score == best.compute_score
                                    && index > best_index)))
                }
            };

            if is_better {
                selected_index = Some(index);
            }
        }

        let Some(index) = selected_index else {
            return Err(ProtocolError::InvalidShardPlan {
                reason: "failed to rebalance weighted shard assignments",
            });
        };
        states[index].assigned_layers -= 1;
        assigned_total -= 1;
    }

    while assigned_total < total_layers {
        let mut selected_index: Option<usize> = None;
        for (index, state) in states.iter().enumerate() {
            if state.assigned_layers >= state.max_layers {
                continue;
            }

            let state_deviation = state.deviation_numerator(total_compute_i128);
            let is_better = match selected_index {
                None => true,
                Some(best_index) => {
                    let best = &states[best_index];
                    let best_deviation = best.deviation_numerator(total_compute_i128);

                    let state_needs_more = state_deviation > 0;
                    let best_needs_more = best_deviation > 0;

                    if state_needs_more != best_needs_more {
                        state_needs_more
                    } else if state_needs_more {
                        state_deviation > best_deviation
                            || (state_deviation == best_deviation
                                && (state.compute_score > best.compute_score
                                    || (state.compute_score == best.compute_score
                                        && index < best_index)))
                    } else {
                        let state_denominator = state.assigned_layers as u128 + 1;
                        let best_denominator = best.assigned_layers as u128 + 1;
                        let state_ratio_numerator =
                            (state.compute_score as u128).saturating_mul(best_denominator);
                        let best_ratio_numerator =
                            (best.compute_score as u128).saturating_mul(state_denominator);

                        state_ratio_numerator > best_ratio_numerator
                            || (state_ratio_numerator == best_ratio_numerator && index < best_index)
                    }
                }
            };

            if is_better {
                selected_index = Some(index);
            }
        }

        let Some(index) = selected_index else {
            return Err(ProtocolError::InvalidShardPlan {
                reason: "insufficient RAM capacity for requested layer count",
            });
        };
        states[index].assigned_layers += 1;
        assigned_total += 1;
    }

    let mut cursor = 0usize;
    let mut assignments = Vec::with_capacity(states.len().min(total_layers));
    for state in states {
        let start = cursor;
        let end = start + state.assigned_layers;
        cursor = end;
        if state.assigned_layers == 0 {
            continue;
        }

        assignments.push(StaticShardAssignment {
            participant: state.participant,
            layer_range: start..end,
        });
    }

    debug_assert_eq!(cursor, total_layers);
    Ok(assignments)
}

pub fn split_layers(total_layers: u32) -> Result<(LayerRange, LayerRange), ProtocolError> {
    if total_layers < 2 {
        return Err(ProtocolError::InvalidShardPlan {
            reason: "at least two layers are required for a two-node split",
        });
    }

    let split_at = total_layers.div_ceil(2);
    let master = LayerRange::new(0, split_at)?;
    let worker = LayerRange::new(split_at, total_layers)?;

    Ok((master, worker))
}

pub fn worker_shard_plan(
    total_layers: u32,
    dtype: DType,
    max_sequence_len: usize,
) -> Result<ShardPlan, ProtocolError> {
    let (_, worker) = split_layers(total_layers)?;
    let plan = ShardPlan {
        total_layers,
        layer_start: worker.start,
        layer_end: worker.end,
        dtype,
        max_sequence_len,
    };
    plan.validate()?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_full_contiguous_coverage(
        assignments: &[StaticShardAssignment<&str>],
        total_layers: usize,
    ) {
        let mut cursor = 0usize;

        for assignment in assignments {
            assert_eq!(assignment.layer_range.start, cursor);
            assert!(assignment.layer_range.start < assignment.layer_range.end);
            cursor = assignment.layer_range.end;
        }

        assert_eq!(cursor, total_layers);
    }

    #[test]
    fn planner_exactly_covers_total_layers() {
        let assignments =
            plan_static_shards(17, &["master", "worker-a", "worker-b"]).expect("valid plan");

        assert_full_contiguous_coverage(&assignments, 17);
    }

    #[test]
    fn planner_is_deterministic_in_input_order() {
        let participants = ["master", "worker-z", "worker-a", "worker-b"];
        let first = plan_static_shards(10, &participants).expect("valid plan");
        let second = plan_static_shards(10, &participants).expect("valid plan");

        assert_eq!(first, second);
        assert_eq!(
            first
                .iter()
                .map(|assignment| assignment.participant)
                .collect::<Vec<_>>(),
            vec!["master", "worker-z", "worker-a", "worker-b"]
        );
    }

    #[test]
    fn planner_splits_even_and_odd_layer_counts() {
        let even = plan_static_shards(8, &["master", "worker-a", "worker-b"]).expect("valid plan");
        assert_eq!(
            even,
            vec![
                StaticShardAssignment {
                    participant: "master",
                    layer_range: 0..3,
                },
                StaticShardAssignment {
                    participant: "worker-a",
                    layer_range: 3..6,
                },
                StaticShardAssignment {
                    participant: "worker-b",
                    layer_range: 6..8,
                },
            ]
        );

        let odd = plan_static_shards(7, &["master", "worker-a", "worker-b"]).expect("valid plan");
        assert_eq!(
            odd,
            vec![
                StaticShardAssignment {
                    participant: "master",
                    layer_range: 0..3,
                },
                StaticShardAssignment {
                    participant: "worker-a",
                    layer_range: 3..5,
                },
                StaticShardAssignment {
                    participant: "worker-b",
                    layer_range: 5..7,
                },
            ]
        );
    }

    #[test]
    fn planner_omits_empty_assignments_when_nodes_exceed_layers() {
        let assignments = plan_static_shards(
            3,
            &["master", "worker-a", "worker-b", "worker-c", "worker-d"],
        )
        .expect("valid plan");

        assert_eq!(
            assignments,
            vec![
                StaticShardAssignment {
                    participant: "master",
                    layer_range: 0..1,
                },
                StaticShardAssignment {
                    participant: "worker-a",
                    layer_range: 1..2,
                },
                StaticShardAssignment {
                    participant: "worker-b",
                    layer_range: 2..3,
                },
            ]
        );
        assert_full_contiguous_coverage(&assignments, 3);
    }

    #[test]
    fn planner_assigns_all_layers_to_master_when_no_workers() {
        let assignments = plan_static_shards(12, &["master"]).expect("valid plan");

        assert_eq!(
            assignments,
            vec![StaticShardAssignment {
                participant: "master",
                layer_range: 0..12,
            }]
        );
    }

    #[test]
    fn planner_rejects_zero_total_layers() {
        assert!(matches!(
            plan_static_shards(0, &["master", "worker-a"]),
            Err(ProtocolError::InvalidShardPlan { .. })
        ));
    }

    #[test]
    fn weighted_planner_exactly_covers_total_layers() {
        let assignments = plan_weighted_shards(
            17,
            &[
                WeightedShardParticipant {
                    participant: "master",
                    capability: NodeCapability {
                        compute_score: 4,
                        available_ram_mb: 4_096,
                    },
                },
                WeightedShardParticipant {
                    participant: "worker-a",
                    capability: NodeCapability {
                        compute_score: 2,
                        available_ram_mb: 4_096,
                    },
                },
                WeightedShardParticipant {
                    participant: "worker-b",
                    capability: NodeCapability {
                        compute_score: 1,
                        available_ram_mb: 4_096,
                    },
                },
            ],
        )
        .expect("weighted planner should produce a valid plan");

        assert_full_contiguous_coverage(&assignments, 17);
    }

    #[test]
    fn weighted_planner_is_proportional_when_unconstrained() {
        let assignments = plan_weighted_shards(
            10,
            &[
                WeightedShardParticipant {
                    participant: "master",
                    capability: NodeCapability {
                        compute_score: 5,
                        available_ram_mb: 4_096,
                    },
                },
                WeightedShardParticipant {
                    participant: "worker-a",
                    capability: NodeCapability {
                        compute_score: 3,
                        available_ram_mb: 4_096,
                    },
                },
                WeightedShardParticipant {
                    participant: "worker-b",
                    capability: NodeCapability {
                        compute_score: 2,
                        available_ram_mb: 4_096,
                    },
                },
            ],
        )
        .expect("weighted planner should produce a valid plan");

        assert_eq!(
            assignments,
            vec![
                StaticShardAssignment {
                    participant: "master",
                    layer_range: 0..5,
                },
                StaticShardAssignment {
                    participant: "worker-a",
                    layer_range: 5..8,
                },
                StaticShardAssignment {
                    participant: "worker-b",
                    layer_range: 8..10,
                },
            ]
        );
    }

    #[test]
    fn weighted_planner_enforces_ram_caps_and_rebalances_layers() {
        let assignments = plan_weighted_shards(
            12,
            &[
                WeightedShardParticipant {
                    participant: "master",
                    capability: NodeCapability {
                        compute_score: 6,
                        available_ram_mb: 200,
                    },
                },
                WeightedShardParticipant {
                    participant: "worker-a",
                    capability: NodeCapability {
                        compute_score: 3,
                        available_ram_mb: 100,
                    },
                },
                WeightedShardParticipant {
                    participant: "worker-b",
                    capability: NodeCapability {
                        compute_score: 1,
                        available_ram_mb: 1_000,
                    },
                },
                WeightedShardParticipant {
                    participant: "worker-c",
                    capability: NodeCapability {
                        compute_score: 1,
                        available_ram_mb: 20,
                    },
                },
            ],
        )
        .expect("weighted planner should rebalance constrained nodes");

        assert_eq!(
            assignments,
            vec![
                StaticShardAssignment {
                    participant: "master",
                    layer_range: 0..4,
                },
                StaticShardAssignment {
                    participant: "worker-a",
                    layer_range: 4..6,
                },
                StaticShardAssignment {
                    participant: "worker-b",
                    layer_range: 6..12,
                },
            ]
        );
        assert_full_contiguous_coverage(&assignments, 12);
        assert!(
            !assignments
                .iter()
                .any(|assignment| assignment.participant == "worker-c")
        );
    }

    #[test]
    fn weighted_planner_is_deterministic_for_same_input() {
        let participants = [
            WeightedShardParticipant {
                participant: "master",
                capability: NodeCapability {
                    compute_score: 11,
                    available_ram_mb: 4_096,
                },
            },
            WeightedShardParticipant {
                participant: "worker-a",
                capability: NodeCapability {
                    compute_score: 9,
                    available_ram_mb: 4_096,
                },
            },
            WeightedShardParticipant {
                participant: "worker-b",
                capability: NodeCapability {
                    compute_score: 7,
                    available_ram_mb: 4_096,
                },
            },
            WeightedShardParticipant {
                participant: "worker-c",
                capability: NodeCapability {
                    compute_score: 5,
                    available_ram_mb: 4_096,
                },
            },
        ];

        let first = plan_weighted_shards(23, &participants).expect("first run should succeed");
        let second = plan_weighted_shards(23, &participants).expect("second run should succeed");

        assert_eq!(first, second);
    }

    #[test]
    fn splits_even_layer_count() {
        let (master, worker) = split_layers(80).expect("valid split");
        assert_eq!(master, LayerRange { start: 0, end: 40 });
        assert_eq!(worker, LayerRange { start: 40, end: 80 });
    }

    #[test]
    fn gives_master_extra_layer_for_odd_count() {
        let (master, worker) = split_layers(5).expect("valid split");
        assert_eq!(master, LayerRange { start: 0, end: 3 });
        assert_eq!(worker, LayerRange { start: 3, end: 5 });
    }

    #[test]
    fn rejects_single_layer_split() {
        assert!(matches!(
            split_layers(1),
            Err(ProtocolError::InvalidShardPlan { .. })
        ));
    }
}
