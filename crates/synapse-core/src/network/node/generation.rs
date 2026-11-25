use std::{collections::HashSet, sync::Arc, time::Instant};

use libp2p::{PeerId, request_response};
use tracing::{info, warn};

use super::{
    ActivationPass, BatchPromptRequest, ChatEvent, GenerationRoutePlan, InFlightActivationHop,
    InFlightRequest, MAX_BATCH_SIZE, SpeculativeRound, StopReason, SynapseNode, SynapseRole,
    WorkerRouteHop,
};
use crate::inference::{ToyLlama, payload_to_tensor, tensor_to_payload};
use crate::network::node::speculation::MAX_DRAFT_LEN;
#[cfg(test)]
use crate::{WeightedShardParticipant, plan_weighted_shards};

#[derive(Clone, Debug, Eq, PartialEq)]
enum RouteParticipant {
    Master,
    Worker(PeerId),
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct RouteWorkerCandidate {
    peer_id: PeerId,
    capability: Option<crate::NodeCapability>,
    /// Phase 2: structured `WorkerCapability` from a heartbeat.
    /// When `Some`, the master uses the capability-aware planner.
    wire_capability: Option<crate::WorkerCapability>,
}

fn activation_tokens_for_position(
    token_ids: &[u32],
    sequence_position: usize,
) -> Result<(ActivationPass, Vec<u32>), String> {
    if token_ids.is_empty() {
        return Err("generation requires at least one token".to_owned());
    }

    let pass = ActivationPass::for_sequence_position(sequence_position);
    let tokens = match pass {
        ActivationPass::Prefill => token_ids.to_vec(),
        ActivationPass::Decode => vec![
            *token_ids
                .last()
                .ok_or_else(|| "decode requires at least one token".to_owned())?,
        ],
        ActivationPass::Speculative => {
            return Err("speculative passes build their own token block".to_owned());
        }
    };

    Ok((pass, tokens))
}

#[cfg(test)]
fn build_generation_route(
    total_layers: usize,
    master_capability: Option<crate::NodeCapability>,
    workers_in_order: &[RouteWorkerCandidate],
) -> Result<GenerationRoutePlan, String> {
    let master = master_capability.ok_or("master memory capacity is unknown")?;
    let mut seen = HashSet::new();
    let mut participants = vec![WeightedShardParticipant {
        participant: RouteParticipant::Master,
        capability: master,
    }];
    for worker in workers_in_order {
        if !seen.insert(worker.peer_id) {
            return Err("duplicate worker in route candidates".to_owned());
        }
        let capability = worker
            .wire_capability
            .as_ref()
            .map(|wire| crate::NodeCapability {
                compute_score: wire.compute_score,
                available_ram_mb: wire.available_ram_mb,
            })
            .or(worker.capability)
            .ok_or("worker memory capacity is unknown")?;
        participants.push(WeightedShardParticipant {
            participant: RouteParticipant::Worker(worker.peer_id),
            capability,
        });
    }
    // One plan covers the full model, including the coordinator. Never shift
    // a complete worker plan by a second independently calculated prefix.
    let assignments = plan_weighted_shards(total_layers, &participants)
        .map_err(|error| format!("no feasible generation route: {error}"))?;
    crate::shard::validate_exact_coverage(
        total_layers,
        &assignments
            .iter()
            .map(|assignment| assignment.layer_range.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(|error| error.to_string())?;

    let mut master_layer_range = None;
    let mut worker_hops = Vec::new();

    for assignment in assignments {
        match assignment.participant {
            RouteParticipant::Master => {
                master_layer_range = Some(assignment.layer_range);
            }
            RouteParticipant::Worker(peer_id) => worker_hops.push(WorkerRouteHop {
                peer_id,
                layer_range: assignment.layer_range,
            }),
        }
    }

    let master_layer_range = master_layer_range.unwrap_or(0..0);

    Ok(GenerationRoutePlan {
        master_layer_range,
        worker_hops,
    })
}

fn ensure_worker_available(
    peer_id: PeerId,
    available_workers: &HashSet<PeerId>,
) -> Result<(), String> {
    if available_workers.contains(&peer_id) {
        return Ok(());
    }
    Err(format!("expected worker {peer_id} unavailable"))
}

impl SynapseNode {
    fn plan_model_route(&self) -> anyhow::Result<GenerationRoutePlan> {
        use crate::inference::pipeline::TensorRoles;
        use crate::shard::{BoundedShardParticipant, plan_bounded_shards};
        let profile = self
            .memory_profile
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("memory profile unavailable"))?;
        let context = self
            .token_ids
            .len()
            .checked_add(self.current_max_tokens())
            .ok_or_else(|| anyhow::anyhow!("context overflow"))?;
        let local_capacity =
            profile.layer_capacity(self.memory_limit_bytes, context, TensorRoles::COORDINATOR)?;
        let mut participants = vec![BoundedShardParticipant {
            participant: RouteParticipant::Master,
            compute_score: self
                .measured_compute_score
                .ok_or_else(|| anyhow::anyhow!("local throughput unmeasured"))?,
            max_layers: if self.coordinator_only {
                0
            } else {
                local_capacity
            },
        }];
        let mut seen = HashSet::new();
        for peer in &self.worker_peers {
            anyhow::ensure!(seen.insert(*peer), "duplicate route candidate");
            if !self.authenticated_workers.contains(peer) {
                continue;
            }
            let cap = self
                .worker_registry
                .capability_for(peer)
                .ok_or_else(|| anyhow::anyhow!("worker capability unknown"))?;
            let bytes = cap
                .available_memory_bytes
                .ok_or_else(|| anyhow::anyhow!("worker memory unknown"))?;
            let max_layers = profile
                .layer_capacity(bytes, context, TensorRoles::WORKER)
                .unwrap_or(0);
            let compute_score = cap.compute_score.unwrap_or(0);
            if max_layers > 0 && compute_score > 0 {
                participants.push(BoundedShardParticipant {
                    participant: RouteParticipant::Worker(*peer),
                    compute_score,
                    max_layers,
                });
            }
        }
        let assignments = plan_bounded_shards(self.total_layers, &participants)?;
        let mut route = GenerationRoutePlan {
            master_layer_range: 0..0,
            worker_hops: Vec::new(),
        };
        for assignment in assignments {
            match assignment.participant {
                RouteParticipant::Master => route.master_layer_range = assignment.layer_range,
                RouteParticipant::Worker(peer_id) => route.worker_hops.push(WorkerRouteHop {
                    peer_id,
                    layer_range: assignment.layer_range,
                }),
            }
        }
        Ok(route)
    }

    /// Called when a new prompt arrives from the CLI REPL channel.
    pub(super) fn handle_new_prompt(&mut self, prompt: String) {
        let tokenizer = match &self.tokenizer {
            Some(tokenizer) => Arc::clone(tokenizer),
            None => {
                self.fail_generation("tokenizer not loaded");
                return;
            }
        };

        // Tokenize the prompt (already formatted with chat template by CLI)
        let ids = match tokenizer.encode(&prompt, false) {
            Ok(ids) => ids,
            Err(error) => {
                self.fail_generation(format!("failed to tokenize prompt: {error}"));
                return;
            }
        };

        info!(
            token_count = ids.len(),
            "prompt tokenized — starting generation"
        );

        self.begin_generation_from_ids(ids);
    }

    /// Shared generation setup used by both single-prompt and batch paths.
    fn begin_generation_from_ids(&mut self, ids: Vec<u32>) {
        self.generation_active = true;

        let context_limit = self
            .model_config
            .as_ref()
            .map_or(0, |config| config.context_limit());
        if ids.is_empty()
            || ids
                .len()
                .checked_add(self.current_max_tokens())
                .is_none_or(|len| len > context_limit)
        {
            self.fail_generation("empty prompt or prompt/output exceeds model context limit");
            return;
        }
        self.release_generation_assignments();

        // Reset draft cache for new conversation turn
        self.draft_cache.clear();
        self.spec_round = None;
        self.kv_cache_len = 0;
        self.awaiting_truncate_ack = false;
        self.pending_truncate_trace_id = None;
        self.pending_truncate_peers.clear();

        self.token_ids = ids;
        self.tokens_generated = 0;
        self.awaiting_clear_cache_ack = false;
        self.pending_cache_reset_trace_id = None;
        self.pending_cache_reset_peers.clear();
        self.in_flight_activation_hop = None;

        let route_plan = match self.plan_model_route() {
            Ok(plan) => plan,
            Err(error) => {
                self.fail_generation(error.to_string());
                return;
            }
        };
        self.split_point = route_plan.master_layer_range.end;
        self.speculative_enabled = route_plan.worker_hops.is_empty();
        self.generation_route_plan = Some(route_plan.clone());

        if let Err(error) = self.begin_route_preparation(&route_plan) {
            self.fail_generation(format!("failed to prepare complete route: {error}"));
        }
    }

    pub(super) fn send_clear_cache_requests(&mut self, peers: &[PeerId]) {
        if peers.is_empty() {
            self.awaiting_clear_cache_ack = false;
            self.pending_cache_reset_trace_id = None;
            self.pending_cache_reset_peers.clear();
            return;
        }

        self.cache_reset_nonce = self.cache_reset_nonce.wrapping_add(1);
        let Some(key) = &self.generation_lease else {
            self.fail_generation("missing session lease");
            return;
        };
        let trace_id = format!("{}c{}", key.trace_prefix(), self.cache_reset_nonce);

        self.awaiting_clear_cache_ack = true;
        self.pending_cache_reset_trace_id = Some(trace_id.clone());
        self.pending_cache_reset_peers = peers.iter().copied().collect();

        for peer in peers {
            info!(%peer, trace_id, "master requesting worker cache reset");
            if let Err(error) = self.send_tracked_request(
                *peer,
                super::super::RequestMessage::ClearCache(crate::ClearCache {
                    trace_id: trace_id.clone(),
                }),
            ) {
                self.fail_generation(error.to_string());
                return;
            }
        }
    }

    /// Run master layers on current token_ids and route ActivationRequest hops via workers.
    pub(super) fn send_next_activation(&mut self) {
        if self.shard_readiness.is_some() {
            return;
        }
        if self.awaiting_clear_cache_ack {
            info!(
                pending_peers = self.pending_cache_reset_peers.len(),
                "waiting for all ClearCacheAck messages before first activation"
            );
            return;
        }

        if self.awaiting_truncate_ack {
            info!(
                pending_peers = self.pending_truncate_peers.len(),
                "waiting for all KvTruncateAck messages before next activation"
            );
            return;
        }

        let pipeline = match &self.pipeline {
            Some(pipeline) => Arc::clone(pipeline),
            None => {
                self.fail_generation("model not loaded");
                return;
            }
        };

        let route_plan = match self.generation_route_plan.clone() {
            Some(plan) => plan,
            None => {
                self.fail_generation("generation route plan is missing");
                return;
            }
        };

        // Phase 4: start timing this step's activation chain so we can
        // compute each worker's share of total step latency once it
        // completes (see `publish_step_telemetry`).
        self.step_started_at = Some(Instant::now());
        self.step_hop_durations_ms.clear();

        // Decide what this step feeds through the route:
        // - step 0 → full prefill
        // - later steps → either a multi-token speculative verification
        //   block (when the draft cache has a prediction) or a plain
        //   single-token decode.
        let (activation_pass, activation_tokens) = if self.tokens_generated == 0 {
            match activation_tokens_for_position(&self.token_ids, 0) {
                Ok(activation) => activation,
                Err(error) => {
                    self.fail_generation(error);
                    return;
                }
            }
        } else {
            match self.build_speculative_activation() {
                Some((tokens, round)) => {
                    self.spec_round = Some(round);
                    (ActivationPass::Speculative, tokens)
                }
                None => {
                    match activation_tokens_for_position(&self.token_ids, self.tokens_generated) {
                        Ok(activation) => activation,
                        Err(error) => {
                            self.fail_generation(error);
                            return;
                        }
                    }
                }
            }
        };

        info!(
            seq_len = activation_tokens.len(),
            step = self.tokens_generated,
            mode = activation_pass.as_str(),
            "master running embed + layers"
        );

        // Create token tensor.
        let token_ids =
            match candle_core::Tensor::new(activation_tokens.as_slice(), &candle_core::Device::Cpu)
            {
                Ok(tensor) => match tensor.reshape((1, activation_tokens.len())) {
                    Ok(tensor) => tensor,
                    Err(error) => {
                        self.fail_generation(format!("failed to reshape token IDs: {error}"));
                        return;
                    }
                },
                Err(error) => {
                    self.fail_generation(format!("failed to create token ID tensor: {error}"));
                    return;
                }
            };

        // Embed + master layers.
        let hidden = match pipeline.embed(&token_ids) {
            Ok(hidden) => hidden,
            Err(error) => {
                self.fail_generation(format!("embedding failed: {error}"));
                return;
            }
        };

        let after_master = match activation_pass {
            ActivationPass::Prefill => {
                pipeline.forward_layers_prefill(&hidden, route_plan.master_layer_range.clone())
            }
            ActivationPass::Decode => {
                pipeline.forward_layers_decode(&hidden, route_plan.master_layer_range.clone())
            }
            ActivationPass::Speculative => {
                pipeline.forward_layers_speculative(&hidden, route_plan.master_layer_range.clone())
            }
        };
        let after_master = match after_master {
            Ok(hidden) => hidden,
            Err(error) => {
                self.fail_generation(format!(
                    "master {} layer forward failed: {error}",
                    activation_pass.as_str()
                ));
                return;
            }
        };

        // Every KV cache along the route now holds these positions too —
        // track the count so speculative rollbacks know the base length.
        self.kv_cache_len += activation_tokens.len();

        // Local-only fast path: when no workers are in the route, the
        // master's hidden state goes straight into lm_head. Serializing it to
        // a `TensorPayload` and immediately deserializing it back (as the old
        // path did) cost two full-tensor f32 copies per token for nothing.
        if route_plan.worker_hops.is_empty() {
            self.in_flight_activation_hop = None;
            self.process_final_activation_output(None, after_master);
            return;
        }

        // Preserve dtype exactly. Lossy activation compression is not a
        // correctness-preserving default and needs separate quality evidence.
        let wire_hidden = after_master;

        // Serialize for network transfer (only needed when a worker hop exists).
        let payload = match tensor_to_payload(&wire_hidden) {
            Ok(payload) => payload,
            Err(error) => {
                self.fail_generation(format!("failed to serialize hidden state: {error}"));
                return;
            }
        };

        self.dispatch_worker_hop(
            &route_plan,
            activation_pass,
            self.tokens_generated,
            0,
            payload,
        );
    }

    /// Build the multi-token verification block for a speculative step.
    ///
    /// Returns `[last_committed_token, d1, ..., dk]` plus the round
    /// bookkeeping when the draft cache predicts at least one token for the
    /// current context; `None` falls back to a plain single-token decode.
    fn build_speculative_activation(&mut self) -> Option<(Vec<u32>, SpeculativeRound)> {
        if !self.speculative_enabled || self.token_ids.len() < 2 {
            return None;
        }

        let draft_tokens = self.draft_cache.draft(&self.token_ids, MAX_DRAFT_LEN);
        if draft_tokens.is_empty() {
            return None;
        }

        let last_token = *self.token_ids.last()?;
        let cache_base_len = self.kv_cache_len;

        let mut tokens = Vec::with_capacity(draft_tokens.len() + 1);
        tokens.push(last_token);
        tokens.extend_from_slice(&draft_tokens);

        Some((
            tokens,
            SpeculativeRound {
                draft_tokens,
                cache_base_len,
            },
        ))
    }

    fn dispatch_worker_hop(
        &mut self,
        route_plan: &GenerationRoutePlan,
        activation_pass: ActivationPass,
        sequence_position: usize,
        hop_index: usize,
        payload: crate::TensorPayload,
    ) {
        let Some(hop) = route_plan.worker_hops.get(hop_index) else {
            self.fail_generation(format!(
                "missing worker hop {hop_index} in generation route"
            ));
            return;
        };

        if let Err(error) = ensure_worker_available(hop.peer_id, &self.authenticated_workers) {
            self.fail_generation(format!("{error} for hop {hop_index}"));
            return;
        }

        // Phase 6: record that this hop consumes a pooled connection.
        self.connection_pool.mark_connection_used(&hop.peer_id);

        let Some(key) = &self.generation_lease else {
            self.fail_generation("missing session lease");
            return;
        };
        let trace_id = format!("{}g{}h{}", key.trace_prefix(), sequence_position, hop_index);
        match activation_pass {
            ActivationPass::Prefill => {
                info!(
                    peer = %hop.peer_id,
                    hop = hop_index,
                    layer_range = ?hop.layer_range,
                    shape = ?payload.shape,
                    original_bytes = payload.original_size,
                    compressed_bytes = payload.compressed_size,
                    ratio = payload.compression_ratio(),
                    codec = ?payload.compression,
                    "master routing prefill hidden state to worker"
                );
            }
            ActivationPass::Decode => {
                info!(
                    peer = %hop.peer_id,
                    hop = hop_index,
                    layer_range = ?hop.layer_range,
                    shape = ?payload.shape,
                    original_bytes = payload.original_size,
                    compressed_bytes = payload.compressed_size,
                    ratio = payload.compression_ratio(),
                    codec = ?payload.compression,
                    "master routing decode hidden state to worker"
                );
            }
            ActivationPass::Speculative => {
                info!(
                    peer = %hop.peer_id,
                    hop = hop_index,
                    layer_range = ?hop.layer_range,
                    shape = ?payload.shape,
                    original_bytes = payload.original_size,
                    compressed_bytes = payload.compressed_size,
                    ratio = payload.compression_ratio(),
                    codec = ?payload.compression,
                    "master routing speculative verification block to worker"
                );
            }
        }

        let request = crate::ActivationRequest {
            trace_id: trace_id.clone(),
            sequence_position,
            layer_range: (hop.layer_range.start, hop.layer_range.end),
            hidden_state: payload,
        };

        self.in_flight_activation_hop = Some(InFlightActivationHop {
            sequence_position,
            hop_index,
            trace_id,
            dispatch_time: Some(Instant::now()),
            pass: activation_pass,
        });

        if let Err(error) = self.send_tracked_request(
            hop.peer_id,
            super::super::RequestMessage::ActivationRequest(request),
        ) {
            self.fail_generation(error.to_string());
        }
    }

    /// Phase 4: publish live per-peer telemetry (layers handled, measured
    /// pace, share of total step latency) for the generation step that
    /// just finished. No-op when no worker hops were involved (local-only
    /// execution, or a step where no hop response was recorded).
    fn publish_step_telemetry(&mut self) {
        let hop_durations = std::mem::take(&mut self.step_hop_durations_ms);
        if hop_durations.is_empty() {
            self.step_started_at = None;
            return;
        }

        let total_step_ms = self
            .step_started_at
            .take()
            .map(|started| started.elapsed().as_millis() as u64)
            .unwrap_or(0)
            .max(1);

        let Some(route_plan) = self.generation_route_plan.clone() else {
            return;
        };

        for hop in &route_plan.worker_hops {
            let Some(&hop_ms) = hop_durations.get(&hop.peer_id) else {
                continue;
            };
            let layers_assigned = hop.layer_range.len() as u32;
            let tokens_per_sec = if hop_ms > 0 {
                1000.0 / hop_ms as f32
            } else {
                0.0
            };
            let util_pct =
                ((hop_ms as f64 / total_step_ms as f64) * 100.0).clamp(0.0, 100.0) as f32;
            self.worker_registry.update_telemetry(
                hop.peer_id,
                layers_assigned,
                tokens_per_sec,
                util_pct,
            );
        }
    }

    fn process_final_activation_output(
        &mut self,
        peer: Option<PeerId>,
        worker_output: candle_core::Tensor,
    ) {
        self.publish_step_telemetry();

        // Phase 10: speculative rounds verify a whole draft block instead
        // of sampling one token.
        if let Some(round) = self.spec_round.take() {
            self.process_speculative_output(peer, worker_output, round);
            return;
        }

        info!(
            peer = ?peer,
            shape = ?worker_output.dims(),
            "master received final activation hidden state — running lm_head"
        );

        let pipeline = match &self.pipeline {
            Some(pipeline) => Arc::clone(pipeline),
            None => {
                self.fail_generation("model not loaded");
                return;
            }
        };

        let logits = match pipeline.head_last_token(&worker_output) {
            Ok(logits) => logits,
            Err(error) => {
                self.fail_generation(format!("lm_head inference failed: {error}"));
                return;
            }
        };

        let next_token = match ToyLlama::sample_argmax(&logits) {
            Ok(token) => token,
            Err(error) => {
                self.fail_generation(format!("sampling failed: {error}"));
                return;
            }
        };

        let tokenizer = match &self.tokenizer {
            Some(tokenizer) => Arc::clone(tokenizer),
            None => {
                self.fail_generation("tokenizer not loaded");
                return;
            }
        };

        let decoded = match tokenizer.decode(&[next_token], true) {
            Ok(text) => text,
            Err(error) => {
                warn!(peer = ?peer, %error, token_id = next_token, "token decode failed");
                format!("<tok:{next_token}>")
            }
        };
        self.emit(ChatEvent::TokenGenerated {
            token_id: next_token,
            position: self.tokens_generated,
        });
        self.emit(ChatEvent::Token(decoded));
        if !self.generation_active {
            return;
        }

        // Feed the N-gram draft cache for speculative decoding.
        if self.speculative_enabled && self.token_ids.len() >= 2 {
            self.draft_cache
                .observe(&self.token_ids[..self.token_ids.len()], next_token);
        }

        self.token_ids.push(next_token);
        self.tokens_generated += 1;
        self.in_flight_activation_hop = None;

        let max_tokens = self.current_max_tokens();
        let should_stop = self.tokens_generated >= max_tokens || tokenizer.is_eos(next_token);

        if should_stop {
            let reason = if tokenizer.is_eos(next_token) {
                StopReason::Eos
            } else {
                StopReason::MaxTokens
            };
            self.finish_generation(reason);
        } else {
            self.local_step_pending = self.generation_active;
        }
    }

    /// Phase 10: verify a speculative draft block against the model's own
    /// logits and commit only the verified prefix plus one correction token.
    ///
    /// Cache bookkeeping: the verification block appended
    /// `1 + draft_len` positions to every KV cache along the route. The
    /// true committed sequence is `accepted` drafts + the last committed
    /// token already present as the block head, so caches are truncated to
    /// `cache_base_len + accepted + 1` positions; rejected tail positions
    /// disappear from both the master's local layers and (via KvTruncate)
    /// every worker hop.
    fn process_speculative_output(
        &mut self,
        peer: Option<PeerId>,
        worker_output: candle_core::Tensor,
        round: SpeculativeRound,
    ) {
        info!(
            peer = ?peer,
            shape = ?worker_output.dims(),
            "master received speculative activation — verifying drafts"
        );
        self.in_flight_activation_hop = None;

        let pipeline = match &self.pipeline {
            Some(pipeline) => Arc::clone(pipeline),
            None => {
                self.fail_generation("model not loaded");
                return;
            }
        };
        let tokenizer = match &self.tokenizer {
            Some(tokenizer) => Arc::clone(tokenizer),
            None => {
                self.fail_generation("tokenizer not loaded");
                return;
            }
        };

        // Full-sequence logits over the verification block:
        // [1, 1 + draft_len, vocab].
        let logits = match pipeline.head(&worker_output) {
            Ok(logits) => logits,
            Err(error) => {
                self.fail_generation(format!("lm_head inference failed: {error}"));
                return;
            }
        };

        let verification =
            match crate::network::node::speculation::verify_draft(&round.draft_tokens, &logits) {
                Ok(verification) => verification,
                Err(error) => {
                    self.fail_generation(format!("draft verification failed: {error}"));
                    return;
                }
            };

        let accepted = verification.accepted_count;
        let Some(correction) = verification.correction_token else {
            self.fail_generation("draft verification produced no continuation token");
            return;
        };

        let target_cache_len = round.cache_base_len + accepted + 1;
        let rejected_positions = self.kv_cache_len.saturating_sub(target_cache_len);

        info!(
            draft_len = round.draft_tokens.len(),
            accepted,
            rejected_positions,
            correction_token = correction,
            "speculative verification complete"
        );

        // Roll back the master-local caches when drafts were rejected.
        if rejected_positions > 0
            && let Err(error) = pipeline.truncate_kv_cache(target_cache_len)
        {
            self.fail_generation(format!("KV cache rollback failed: {error}"));
            return;
        }
        self.kv_cache_len = target_cache_len;

        // Commit accepted drafts + the correction token, stopping at EOS or
        // the turn's token budget.
        let committed: Vec<u32> = round.draft_tokens[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(correction))
            .collect();

        let max_tokens = self.current_max_tokens();
        let mut stop_reason: Option<StopReason> = None;

        for &token in &committed {
            if stop_reason.is_some() {
                break;
            }

            // Feed the N-gram draft cache with each newly committed token
            // (context excludes the token itself, matching decode steps).
            if self.speculative_enabled && self.token_ids.len() >= 2 {
                self.draft_cache.observe(&self.token_ids, token);
            }

            let decoded = match tokenizer.decode(&[token], true) {
                Ok(text) => text,
                Err(error) => {
                    warn!(peer = ?peer, %error, token_id = token, "token decode failed");
                    format!("<tok:{token}>")
                }
            };
            self.emit(ChatEvent::TokenGenerated {
                token_id: token,
                position: self.tokens_generated,
            });
            self.emit(ChatEvent::Token(decoded));
            if !self.generation_active {
                return;
            }

            self.token_ids.push(token);
            self.tokens_generated += 1;

            if tokenizer.is_eos(token) {
                stop_reason = Some(StopReason::Eos);
            } else if self.tokens_generated >= max_tokens {
                stop_reason = Some(StopReason::MaxTokens);
            }
        }

        if let Some(reason) = stop_reason {
            self.finish_generation(reason);
            return;
        }

        // Workers still hold rejected positions in their KV caches. Roll
        // them back and wait for acks before feeding the next activation;
        // a local-only route truncates above and continues immediately.
        let has_worker_hops = self
            .generation_route_plan
            .as_ref()
            .is_some_and(|plan| !plan.worker_hops.is_empty());
        if rejected_positions > 0 && has_worker_hops {
            self.send_kv_truncate_requests(target_cache_len);
            return;
        }

        self.local_step_pending = self.generation_active;
    }

    /// Shared end-of-turn bookkeeping for all generation paths.
    fn finish_generation(&mut self, reason: StopReason) {
        self.release_generation_assignments();
        self.emit(ChatEvent::GenerationDone {
            tokens_generated: self.tokens_generated,
            reason,
        });
        self.generation_active = false;
        self.awaiting_clear_cache_ack = false;
        self.pending_cache_reset_trace_id = None;
        self.pending_cache_reset_peers.clear();
        self.generation_route_plan = None;
        self.batch_active_max_tokens = None;
        self.spec_round = None;
        self.awaiting_truncate_ack = false;
        self.pending_truncate_trace_id = None;
        self.pending_truncate_peers.clear();
        // Process next batch request if any are queued.
        self.drain_batch_queue();
    }

    /// Max tokens for the active turn: batch override when set, otherwise
    /// the role default.
    pub(super) fn current_max_tokens(&self) -> usize {
        self.batch_active_max_tokens.unwrap_or(match &self.role {
            SynapseRole::Master { max_tokens, .. } => *max_tokens,
            _ => 50,
        })
    }

    /// Ask every worker hop to roll its KV caches back to `target_seq_len`
    /// positions after rejected speculative drafts. The next activation is
    /// dispatched once all acks arrive (mirrors the ClearCache handshake).
    fn send_kv_truncate_requests(&mut self, target_seq_len: usize) {
        let peers: Vec<PeerId> = match &self.generation_route_plan {
            Some(plan) => plan.worker_hops.iter().map(|hop| hop.peer_id).collect(),
            None => {
                self.fail_generation("missing generation route during KV rollback");
                return;
            }
        };

        if peers.is_empty() {
            self.awaiting_truncate_ack = false;
            self.pending_truncate_trace_id = None;
            self.pending_truncate_peers.clear();
            self.local_step_pending = self.generation_active;
            return;
        }

        self.truncate_nonce = self.truncate_nonce.wrapping_add(1);
        let Some(key) = &self.generation_lease else {
            self.fail_generation("missing session lease");
            return;
        };
        let trace_id = format!("{}t{}", key.trace_prefix(), self.truncate_nonce);

        self.awaiting_truncate_ack = true;
        self.pending_truncate_trace_id = Some(trace_id.clone());
        self.pending_truncate_peers = peers.iter().copied().collect();

        for peer in peers {
            info!(
                %peer,
                trace_id,
                target_seq_len,
                "master requesting worker KV truncation"
            );
            if let Err(error) = self.send_tracked_request(
                peer,
                super::super::RequestMessage::KvTruncate(crate::KvTruncate {
                    trace_id: trace_id.clone(),
                    target_seq_len,
                }),
            ) {
                self.fail_generation(error.to_string());
                return;
            }
        }
    }

    pub(super) fn peer_required_for_active_generation(&self, peer: PeerId) -> bool {
        self.pending_cache_reset_peers.contains(&peer)
            || self
                .generation_route_plan
                .as_ref()
                .is_some_and(|plan| plan.worker_hops.iter().any(|hop| hop.peer_id == peer))
    }

    pub(super) fn handle_activation_request(
        &mut self,
        peer: PeerId,
        request: crate::ActivationRequest,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
    ) {
        if !self.authorize_worker_trace(peer, &request.trace_id) {
            self.reject(
                channel,
                "operation requires the active authenticated session lease",
            );
            return;
        }

        if !matches!(self.role, SynapseRole::Worker { .. }) {
            warn!(%peer, "master received ActivationRequest; rejecting");
            self.reject(channel, "ActivationRequest is handled by workers");
            return;
        }

        if let Err(error) = request.validate() {
            warn!(%peer, %error, "ActivationRequest validation failed");
            self.reject(channel, "invalid ActivationRequest");
            return;
        }

        // Classify the compute pass from the sequence position and the
        // activation's token count. Multi-token blocks at non-zero
        // positions are speculative verification passes that append to the
        // existing KV caches instead of replacing them.
        let activation_seq_len = request.hidden_state.shape.get(1).copied().unwrap_or(0);
        let activation_pass =
            ActivationPass::for_request(request.sequence_position, activation_seq_len);

        info!(
            %peer,
            trace_id = %request.trace_id,
            mode = activation_pass.as_str(),
            layer_range = ?request.layer_range,
            shape = ?request.hidden_state.shape,
            "worker received ActivationRequest — running inference"
        );

        // Deserialize the hidden state tensor.
        let hidden = match payload_to_tensor(&request.hidden_state) {
            Ok(tensor) => tensor,
            Err(error) => {
                warn!(%peer, %error, "failed to deserialize hidden state tensor");
                self.reject(channel, "tensor deserialization failed");
                return;
            }
        };

        // Phase 5: the wire carries bf16; upcast to f32 before running
        // layers — weights are f32 and candle matmul requires matching
        // dtypes on both operands.
        let hidden = match hidden.to_dtype(
            self.pipeline
                .as_ref()
                .map_or(candle_core::DType::F32, |pipeline| pipeline.model_dtype()),
        ) {
            Ok(tensor) => tensor,
            Err(error) => {
                warn!(%peer, %error, "failed to upcast hidden state to f32");
                self.reject(channel, "tensor dtype conversion failed");
                return;
            }
        };

        // Run worker layers.
        let pipeline = match &self.pipeline {
            Some(pipeline) => Arc::clone(pipeline),
            None => {
                warn!(%peer, "model not loaded yet");
                self.reject(channel, "model not loaded");
                return;
            }
        };

        let (layer_start, layer_end) = request.layer_range;
        let Some(active) = &self.worker_assignment else {
            self.reject(channel, "no active assignment");
            return;
        };
        if (layer_start..layer_end) != active.assignment.layer_range()
            || request.hidden_state.shape.first() != Some(&1)
            || activation_seq_len == 0
            || activation_seq_len > active.assignment.context_tokens as usize
        {
            self.reject(
                channel,
                "activation shape or range differs from leased assignment",
            );
            return;
        }

        if let Err(error) = active.cursor.validate_step(
            request.sequence_position,
            activation_seq_len,
            active.assignment.context_tokens as usize,
        ) {
            self.reject(channel, &error.to_string());
            return;
        }

        let result = match activation_pass {
            ActivationPass::Prefill => {
                pipeline.forward_layers_prefill(&hidden, layer_start..layer_end)
            }
            ActivationPass::Decode => {
                pipeline.forward_layers_decode(&hidden, layer_start..layer_end)
            }
            ActivationPass::Speculative => {
                pipeline.forward_layers_speculative(&hidden, layer_start..layer_end)
            }
        };
        let output = match result {
            Ok(tensor) => tensor,
            Err(error) => {
                warn!(
                    %peer,
                    %error,
                    mode = activation_pass.as_str(),
                    "worker inference failed"
                );
                self.reject(channel, "inference failed");
                return;
            }
        };

        let wire_output = output;

        // Serialize the output tensor.
        let payload = match tensor_to_payload(&wire_output) {
            Ok(payload) => payload,
            Err(error) => {
                warn!(%peer, %error, "failed to serialize output tensor");
                self.reject(channel, "tensor serialization failed");
                return;
            }
        };

        match activation_pass {
            ActivationPass::Prefill => {
                info!(
                    %peer,
                    trace_id = %request.trace_id,
                    output_shape = ?payload.shape,
                    "worker prefill inference complete — sending response"
                );
            }
            ActivationPass::Decode => {
                info!(
                    %peer,
                    trace_id = %request.trace_id,
                    output_shape = ?payload.shape,
                    "worker decode inference complete — sending response (expected [1,1,hidden])"
                );
            }
            ActivationPass::Speculative => {
                info!(
                    %peer,
                    trace_id = %request.trace_id,
                    output_shape = ?payload.shape,
                    "worker speculative verification complete — sending multi-token response"
                );
            }
        }

        if let Some(active) = self.worker_assignment.as_mut() {
            active.cursor.commit_step(activation_seq_len);
        }
        let response = crate::ActivationResponse {
            trace_id: request.trace_id,
            logits: payload,
        };

        self.send_response(
            channel,
            super::super::ResponseMessage::ActivationResponse(response),
        );
        // Worker stays alive for more requests — never sets exchange_complete here.
    }

    pub(super) fn handle_clear_cache_request(
        &mut self,
        peer: PeerId,
        request: crate::ClearCache,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
    ) {
        if !self.authorize_worker_trace(peer, &request.trace_id) {
            self.reject(
                channel,
                "operation requires the active authenticated session lease",
            );
            return;
        }

        if !matches!(self.role, SynapseRole::Worker { .. }) {
            warn!(%peer, "master received ClearCache request; rejecting");
            self.reject(channel, "ClearCache is handled by workers");
            return;
        }

        if let Err(error) = request.validate() {
            warn!(%peer, %error, "ClearCache validation failed");
            self.reject(channel, "invalid ClearCache request");
            return;
        }

        let reset = match self
            .worker_assignment
            .as_mut()
            .expect("authorized active lease")
            .cursor
            .reset(&request.trace_id)
        {
            Ok(reset) => reset,
            Err(error) => {
                self.reject(channel, &error.to_string());
                return;
            }
        };
        if !reset {
            self.send_response(
                channel,
                super::super::ResponseMessage::ClearCacheAck(request),
            );
            return;
        }

        let pipeline = match &self.pipeline {
            Some(pipeline) => Arc::clone(pipeline),
            None => {
                warn!(%peer, "model not loaded yet");
                self.reject(channel, "model not loaded");
                return;
            }
        };

        if let Err(error) = pipeline.clear_kv_cache() {
            warn!(%peer, %error, "worker failed to clear KV cache");
            self.reject(channel, "failed to clear KV cache");
            return;
        }

        info!(
            %peer,
            trace_id = %request.trace_id,
            "worker cleared KV cache and acknowledged request"
        );
        self.send_response(
            channel,
            super::super::ResponseMessage::ClearCacheAck(request),
        );
    }

    /// Handle KvTruncate request from the master: roll this worker's owned
    /// KV caches back to `target_seq_len` positions after a speculative
    /// round rejected draft tokens.
    pub(super) fn handle_kv_truncate_request(
        &mut self,
        peer: PeerId,
        request: crate::KvTruncate,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
    ) {
        if !self.authorize_worker_trace(peer, &request.trace_id) {
            self.reject(channel, "unauthorized cache rollback");
            return;
        }
        self.reject(
            channel,
            "distributed speculative rollback is not enabled for protocol v1 sessions",
        );
    }

    /// Handle HealthPing request from master to pre-warm connection.
    /// Workers respond with HealthPong containing current timestamp.
    pub(super) fn handle_health_ping(
        &mut self,
        peer: PeerId,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
    ) {
        if !self.authenticated_workers.contains(&peer) || self.revocations.contains(&peer) {
            self.reject(channel, "unauthorized health probe");
            return;
        }
        info!(%peer, "received HealthPing, responding with HealthPong");
        let health = crate::Health {
            loaded_shard: None,
            latency_ms: Some(0),
            memory_bytes: None,
            last_error: None,
        };
        self.send_response(channel, super::super::ResponseMessage::HealthPong(health));
    }

    pub(super) fn handle_inbound_response(
        &mut self,
        peer: PeerId,
        response: super::super::ResponseMessage,
    ) {
        match response {
            super::super::ResponseMessage::CapabilityAck { accepted } => {
                info!(%peer, accepted, "master acked capability ping");
            }
            super::super::ResponseMessage::ChallengeIssued(challenge) => {
                if !self.greeted_peers.contains(&peer) || self.revocations.contains(&peer) {
                    return;
                }
                let SynapseRole::Worker { pairing_code, .. } = &self.role else {
                    return;
                };
                let response = match crate::pairing::respond_to_challenge(
                    &self.identity,
                    &challenge,
                    pairing_code,
                    peer,
                    super::assignments::unix_now(),
                ) {
                    Ok(response) => response,
                    Err(error) => {
                        warn!(%peer,%error,"invalid coordinator challenge");
                        return;
                    }
                };
                self.pending_coordinator_challenges.insert(peer, challenge);
                if let Err(error) = self.send_tracked_request(
                    peer,
                    super::super::RequestMessage::ChallengeResponse(response),
                ) {
                    warn!(%peer, %error, "pairing response not sent");
                    self.pending_coordinator_challenges.remove(&peer);
                }
            }
            super::super::ResponseMessage::PairingAccepted(proof) => {
                let Some(challenge) = self.pending_coordinator_challenges.remove(&peer) else {
                    return;
                };
                let SynapseRole::Worker { pairing_code, .. } = &self.role else {
                    return;
                };
                if self.revocations.contains(&peer)
                    || crate::pairing::verify_coordinator_proof(
                        &challenge,
                        &proof,
                        pairing_code,
                        peer,
                        self.local_peer_id(),
                        super::assignments::unix_now(),
                    )
                    .is_err()
                {
                    warn!(%peer,"coordinator did not prove possession of pairing secret");
                    let _ = self.swarm.disconnect_peer_id(peer);
                    return;
                }
                self.authenticated_workers.insert(peer);
                self.reset_redial_backoff();
                info!(%peer,"coordinator mutually authenticated");
                let mut capability =
                    crate::WorkerCapability::from_probe(&crate::hardware::HardwareProbe::detect());
                capability.compute_score = self.measured_compute_score.unwrap_or(1);
                capability.available_ram_mb = self.memory_limit_bytes.div_ceil(1024 * 1024);
                if let Err(error) = self.send_tracked_request(
                    peer,
                    super::super::RequestMessage::CapabilityPing(capability),
                ) {
                    warn!(%peer, %error, "capability confirmation not sent");
                }
            }
            super::super::ResponseMessage::ShardReady(ready) => {
                self.handle_shard_ready(peer, ready)
            }
            super::super::ResponseMessage::ShardReleased(_) => {}
            super::super::ResponseMessage::WeightTransfer(_transfer) => {
                // Not implemented in this phase yet
            }
            super::super::ResponseMessage::Authenticated => {
                warn!(%peer,"obsolete unauthenticated acceptance rejected");
            }
            super::super::ResponseMessage::ActivationResponse(response) => {
                if let Err(error) = response.validate() {
                    self.fail_generation(format!(
                        "invalid ActivationResponse from {peer}: {error}"
                    ));
                    return;
                }

                let in_flight = match self.in_flight_activation_hop.clone() {
                    Some(in_flight) => in_flight,
                    None => {
                        self.fail_generation(format!(
                            "unexpected ActivationResponse from {peer} without an in-flight hop"
                        ));
                        return;
                    }
                };
                if in_flight.sequence_position != self.tokens_generated {
                    self.fail_generation(format!(
                        "stale ActivationResponse from {peer}: expected step {}, got {}",
                        self.tokens_generated, in_flight.sequence_position
                    ));
                    return;
                }
                if response.trace_id != in_flight.trace_id {
                    self.fail_generation(format!(
                        "mismatched ActivationResponse trace from {peer}: expected {}, got {}",
                        in_flight.trace_id, response.trace_id
                    ));
                    return;
                }

                let route_plan = match self.generation_route_plan.clone() {
                    Some(plan) => plan,
                    None => {
                        self.fail_generation("missing generation route while processing response");
                        return;
                    }
                };
                let Some(current_hop) = route_plan.worker_hops.get(in_flight.hop_index) else {
                    self.fail_generation(format!(
                        "missing hop {} in generation route",
                        in_flight.hop_index
                    ));
                    return;
                };
                if peer != current_hop.peer_id {
                    self.fail_generation(format!(
                        "ActivationResponse came from unexpected worker {peer}; expected {}",
                        current_hop.peer_id
                    ));
                    return;
                }

                // Phase 4: record how long this hop took (dispatch → response)
                // so `publish_step_telemetry` can compute its live tokens/s
                // and share of the step's total latency once the step ends.
                let hop_duration_ms = in_flight
                    .dispatch_time
                    .map(|dispatched| dispatched.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                self.step_hop_durations_ms.insert(peer, hop_duration_ms);

                // Phase 1 Task 1.5: Record latency for adaptive timeout calculation
                if let Some(dispatch_time) = in_flight.dispatch_time {
                    let latency = dispatch_time.elapsed();
                    // Update per-worker latency window (last 20 samples)
                    self.worker_latencies
                        .entry(peer)
                        .or_insert_with(|| std::collections::VecDeque::with_capacity(20))
                        .push_back(latency);
                    if self.worker_latencies.get(&peer).unwrap().len() > 20 {
                        self.worker_latencies.get_mut(&peer).unwrap().pop_front();
                    }
                    // Update per-hop latency window (last 20 samples)
                    self.hop_latencies
                        .entry(in_flight.hop_index)
                        .or_insert_with(|| std::collections::VecDeque::with_capacity(20))
                        .push_back(latency);
                    if self.hop_latencies.get(&in_flight.hop_index).unwrap().len() > 20 {
                        self.hop_latencies
                            .get_mut(&in_flight.hop_index)
                            .unwrap()
                            .pop_front();
                    }
                }

                // Phase 5: feed per-token throughput into this worker's
                // batch calibration (tokens / second at current batch size).
                let latency_secs = in_flight
                    .dispatch_time
                    .map(|d| d.elapsed().as_secs_f32())
                    .unwrap_or(0.0);
                if latency_secs > 0.0 && in_flight.sequence_position > 0 {
                    let throughput = in_flight.sequence_position as f32 / latency_secs;
                    let recalibrated = self
                        .batch_profiles
                        .entry(peer)
                        .or_insert_with(super::batch_tuning::BatchCalibration::new)
                        .record_sample(throughput);
                    if recalibrated {
                        info!(%peer, throughput, "batch calibration updated");
                    }
                }

                // Phase 6: mark this pooled connection as used + healthy.
                self.connection_pool.mark_connection_used(&peer);
                self.connection_pool.update_peer_health(&peer, true);

                let next_hop_index = in_flight.hop_index + 1;
                if route_plan.worker_hops.get(next_hop_index).is_some() {
                    // The pass travels with the hop so multi-token
                    // speculative blocks stay on the append path across
                    // every worker hop.
                    let activation_pass = in_flight.pass;
                    self.dispatch_worker_hop(
                        &route_plan,
                        activation_pass,
                        in_flight.sequence_position,
                        next_hop_index,
                        response.logits,
                    );
                } else {
                    // Final hop returned over the wire — deserialize once here
                    // before lm_head (the local-only path skips this entirely).
                    // Phase 5: the wire carries bf16; upcast to f32 before
                    // lm_head, which is tied to the f32 embedding weights.
                    let worker_output = match payload_to_tensor(&response.logits).and_then(|t| {
                        t.to_dtype(
                            self.pipeline
                                .as_ref()
                                .map_or(candle_core::DType::F32, |pipeline| pipeline.model_dtype()),
                        )
                        .map_err(Into::into)
                    }) {
                        Ok(tensor) => tensor,
                        Err(error) => {
                            self.fail_generation(format!(
                                "failed to deserialize worker output from {peer}: {error}"
                            ));
                            return;
                        }
                    };
                    self.process_final_activation_output(Some(peer), worker_output);
                }
            }
            super::super::ResponseMessage::ClearCacheAck(ack) => {
                if let Err(error) = ack.validate() {
                    warn!(%peer, %error, "invalid ClearCacheAck from peer");
                    return;
                }

                if !self.generation_active {
                    warn!(%peer, trace_id = %ack.trace_id, "ignoring ClearCacheAck without active generation");
                    return;
                }

                if !self.awaiting_clear_cache_ack {
                    warn!(%peer, trace_id = %ack.trace_id, "unexpected ClearCacheAck received");
                    return;
                }

                let expected_trace = match &self.pending_cache_reset_trace_id {
                    Some(trace_id) => trace_id,
                    None => {
                        warn!(%peer, trace_id = %ack.trace_id, "missing expected cache reset trace id");
                        return;
                    }
                };

                if ack.trace_id != *expected_trace {
                    warn!(
                        %peer,
                        expected_trace_id = %expected_trace,
                        actual_trace_id = %ack.trace_id,
                        "ignoring stale ClearCacheAck"
                    );
                    return;
                }

                if !self.pending_cache_reset_peers.remove(&peer) {
                    warn!(
                        %peer,
                        trace_id = %ack.trace_id,
                        "received ClearCacheAck from peer outside pending cache reset set"
                    );
                    return;
                }

                if !self.pending_cache_reset_peers.is_empty() {
                    info!(
                        pending_peers = self.pending_cache_reset_peers.len(),
                        trace_id = %ack.trace_id,
                        "waiting for remaining cache reset acknowledgements"
                    );
                    return;
                }

                self.awaiting_clear_cache_ack = false;
                self.pending_cache_reset_trace_id = None;
                info!(
                    %peer,
                    trace_id = %ack.trace_id,
                    "cache clear acknowledged — sending first activation"
                );
                if self.healing_in_progress {
                    self.complete_healing();
                } else {
                    self.local_step_pending = self.generation_active;
                }
            }
            super::super::ResponseMessage::KvTruncateAck(ack) => {
                if let Err(error) = ack.validate() {
                    warn!(%peer, %error, "invalid KvTruncateAck from peer");
                    return;
                }

                if !self.generation_active {
                    warn!(%peer, trace_id = %ack.trace_id, "ignoring KvTruncateAck without active generation");
                    return;
                }

                if !self.awaiting_truncate_ack {
                    warn!(%peer, trace_id = %ack.trace_id, "unexpected KvTruncateAck received");
                    return;
                }

                let expected_trace = match &self.pending_truncate_trace_id {
                    Some(trace_id) => trace_id,
                    None => {
                        warn!(%peer, trace_id = %ack.trace_id, "missing expected truncate trace id");
                        return;
                    }
                };

                if ack.trace_id != *expected_trace {
                    warn!(
                        %peer,
                        expected_trace_id = %expected_trace,
                        actual_trace_id = %ack.trace_id,
                        "ignoring stale KvTruncateAck"
                    );
                    return;
                }

                if !self.pending_truncate_peers.remove(&peer) {
                    warn!(
                        %peer,
                        trace_id = %ack.trace_id,
                        "received KvTruncateAck from peer outside pending truncate set"
                    );
                    return;
                }

                if !self.pending_truncate_peers.is_empty() {
                    info!(
                        pending_peers = self.pending_truncate_peers.len(),
                        trace_id = %ack.trace_id,
                        "waiting for remaining KV truncation acknowledgements"
                    );
                    return;
                }

                self.awaiting_truncate_ack = false;
                self.pending_truncate_trace_id = None;
                info!(
                    trace_id = %ack.trace_id,
                    target_seq_len = ack.target_seq_len,
                    "KV truncation acknowledged — resuming generation"
                );
                self.local_step_pending = self.generation_active;
            }
            super::super::ResponseMessage::HealthPong(health) => {
                info!(%peer, latency_ms = ?health.latency_ms, "connection pre-warmed with HealthPong");
            }
            super::super::ResponseMessage::Rejected { reason } => {
                self.fail_generation(format!("request rejected by peer {peer}: {reason}"));
            }
        }
    }
}

// ─── Phase 8: Continuous Batching ────────────────────────────────────────────

impl SynapseNode {
    /// Accept a prompt request from the batch queue (multi-client path).
    ///
    /// Tokenizes the prompt, assigns a request ID, and enqueues it. If no
    /// generation is currently active, starts processing the next queued
    /// request immediately.
    pub(super) fn handle_batch_prompt_request(&mut self, req: BatchPromptRequest) {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let tokenizer = match &self.tokenizer {
            Some(t) => Arc::clone(t),
            None => {
                let _ = req.event_tx.try_send(ChatEvent::Token(
                    "[generation-error] tokenizer not loaded".to_owned(),
                ));
                let _ = req.event_tx.try_send(ChatEvent::GenerationDone {
                    tokens_generated: 0,
                    reason: StopReason::Error,
                });
                return;
            }
        };

        let ids = match tokenizer.encode(&req.prompt, false) {
            Ok(ids) => ids,
            Err(error) => {
                let _ = req.event_tx.try_send(ChatEvent::Token(format!(
                    "[generation-error] tokenization failed: {error}"
                )));
                let _ = req.event_tx.try_send(ChatEvent::GenerationDone {
                    tokens_generated: 0,
                    reason: StopReason::Error,
                });
                return;
            }
        };

        if self.active_batch.len() >= MAX_BATCH_SIZE {
            warn!(
                batch_size = self.active_batch.len(),
                max = MAX_BATCH_SIZE,
                "batch queue full, rejecting prompt"
            );
            let _ = req.event_tx.try_send(ChatEvent::Token(
                "[batch-full] server busy; try again shortly".to_owned(),
            ));
            let _ = req.event_tx.try_send(ChatEvent::GenerationDone {
                tokens_generated: 0,
                reason: StopReason::Error,
            });
            return;
        }

        self.active_batch.push_back(InFlightRequest {
            request_id,
            token_ids: ids,
            max_tokens: req.max_tokens,
            event_tx: req.event_tx,
        });

        info!(
            request_id,
            queue_depth = self.active_batch.len(),
            "batch prompt request queued"
        );

        if !self.generation_active {
            self.drain_batch_queue();
        }
    }

    /// Dequeue the next batch request and start generation for it.
    ///
    /// Installs the per-request event channel so tokens stream back to
    /// the originating client. Reuses the same generation pipeline as
    /// the single-prompt REPL path.
    fn drain_batch_queue(&mut self) {
        if self.active_batch.is_empty() || self.generation_active {
            return;
        }

        let req = self.active_batch.pop_front().expect("batch non-empty");
        info!(
            request_id = req.request_id,
            token_count = req.token_ids.len(),
            remaining = self.active_batch.len(),
            "dequeuing batch request for generation"
        );

        // Install per-request event channel and max_tokens override.
        self.event_tx = Some(req.event_tx);
        self.batch_active_max_tokens = Some(req.max_tokens);

        self.begin_generation_from_ids(req.token_ids);
    }
}

// ─── Phase 9: Self-Healing Swarm ─────────────────────────────────────────────

impl SynapseNode {
    /// Periodic check for timed-out worker hops (called from the run loop).
    ///
    /// If the in-flight activation hop has been waiting longer than
    /// adaptive timeout (or `WORKER_HOP_TIMEOUT` fallback), we assume the
    /// worker is dead and initiate self-healing to reroute through remaining workers.
    pub(super) fn check_worker_timeouts(&mut self) {
        self.expire_assignments();
        if !self.generation_active || self.healing_in_progress {
            return;
        }

        let timed_out_peer = match &self.in_flight_activation_hop {
            Some(hop) => match hop.dispatch_time {
                Some(dispatch_time) => {
                    // Phase 1 Task 1.5: Use adaptive timeout based on observed latency
                    let timeout = self.compute_adaptive_timeout(hop.hop_index);
                    if dispatch_time.elapsed() > timeout {
                        let route_plan = match &self.generation_route_plan {
                            Some(plan) => plan,
                            None => return,
                        };
                        route_plan.worker_hops.get(hop.hop_index).map(|h| h.peer_id)
                    } else {
                        None
                    }
                }
                None => None,
            },
            None => None,
        };

        if let Some(dead_peer) = timed_out_peer {
            warn!(
                %dead_peer,
                "worker hop timed out — initiating self-healing"
            );
            self.initiate_healing(dead_peer);
        }
    }

    /// Begin the self-healing process after a worker disconnect or timeout.
    ///
    /// Removes the dead peer, recomputes the route with remaining workers,
    /// saves current generation state, clears caches on the new route, and
    /// prepares for a fast prefill replay once acks arrive.
    pub(super) fn initiate_healing(&mut self, dead_peer: PeerId) {
        self.authenticated_workers.remove(&dead_peer);
        self.worker_peers.retain(|peer| *peer != dead_peer);
        self.worker_registry.mark_disconnected(dead_peer);
        self.fail_generation(format!("required worker {dead_peer} unavailable; session released without unsafe full-model fallback"));
    }

    /// Complete the self-healing process by replaying the prefill pass.
    ///
    /// Called once all ClearCacheAck messages are received for the new route.
    /// Replays the full token sequence through the new route to rebuild
    /// KV caches, then resumes normal decode from where we left off.
    fn complete_healing(&mut self) {
        let saved_tokens = match self.healing_saved_tokens.take() {
            Some(tokens) => tokens,
            None => {
                warn!("healing complete but no saved tokens — resuming normal generation");
                self.healing_in_progress = false;
                self.local_step_pending = self.generation_active;
                return;
            }
        };

        info!(
            token_count = saved_tokens.len(),
            saved_step = self.healing_saved_step,
            "self-healing: replaying prefill through new route"
        );

        // Restore token state and restart generation from step 0.
        // The prefill pass will rebuild all KV caches.
        self.token_ids = saved_tokens;
        self.tokens_generated = 0;
        self.healing_in_progress = false;

        // Resume normal generation — the first send_next_activation call
        // will do a prefill (tokens_generated == 0), rebuilding KV caches.
        self.local_step_pending = self.generation_active;
    }
}

// ─── Phase 10: Speculative Decoding ──────────────────────────────────────────
//
// The full draft → verify → accept/reject → rollback cycle now runs inside
// the master generation loop:
//
// - `build_speculative_activation` drafts up to MAX_DRAFT_LEN tokens from the
//   N-gram cache and builds the verification block
//   `[last_committed_token, d1..dk]`.
// - The block flows through the normal route as an
//   ActivationPass::Speculative activation; workers append it with
//   forward_layers_speculative (KV-append, never replace).
// - `process_speculative_output` runs lm_head over all block positions and
//   verifies drafts against greedy argmax (speculation::verify_draft).
// - Rejected positions roll back via truncate_kv_cache on the master and
//   KvTruncate requests (ack-gated) on every worker hop.
//
// Cache accounting invariant: after a round the caches hold exactly
// `cache_base_len + accepted + 1` positions; the correction token is fed by
// the following decode step.

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use libp2p::PeerId;

    use super::{
        ActivationPass, RouteWorkerCandidate, activation_tokens_for_position,
        build_generation_route, ensure_worker_available,
    };
    use crate::NodeCapability;

    #[test]
    fn first_step_uses_prefill_tokens() {
        let token_ids = vec![11_u32, 22, 33];
        let (pass, active_tokens) =
            activation_tokens_for_position(&token_ids, 0).expect("prefill selection should work");

        assert_eq!(pass, ActivationPass::Prefill);
        assert_eq!(active_tokens, token_ids);
    }

    #[test]
    fn later_steps_use_single_decode_token() {
        let token_ids = vec![11_u32, 22, 33, 44];
        let (pass, active_tokens) =
            activation_tokens_for_position(&token_ids, 1).expect("decode selection should work");

        assert_eq!(pass, ActivationPass::Decode);
        assert_eq!(active_tokens, vec![44_u32]);
    }

    #[test]
    fn activation_requires_non_empty_tokens() {
        let err =
            activation_tokens_for_position(&[], 0).expect_err("empty token sequence should fail");
        assert!(
            err.contains("requires at least one token"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn activation_pass_classification_from_request_shape() {
        use super::super::ActivationPass;

        // Position 0 is always a cache-replacing prefill.
        assert_eq!(ActivationPass::for_request(0, 7), ActivationPass::Prefill);
        assert_eq!(ActivationPass::for_request(0, 1), ActivationPass::Prefill);

        // Later positions with a single token are plain decode.
        assert_eq!(ActivationPass::for_request(4, 1), ActivationPass::Decode);

        // Later positions with multi-token blocks are speculative.
        assert_eq!(
            ActivationPass::for_request(4, 4),
            ActivationPass::Speculative
        );
    }

    #[test]
    fn multi_hop_route_is_ordered_and_contiguous_end_to_end() {
        let worker_a = PeerId::random();
        let worker_b = PeerId::random();
        let worker_c = PeerId::random();
        let worker_d = PeerId::random();
        let workers = vec![
            RouteWorkerCandidate {
                peer_id: worker_a,
                capability: Some(NodeCapability {
                    compute_score: 1,
                    available_ram_mb: 4096,
                }),
                wire_capability: None,
            },
            RouteWorkerCandidate {
                peer_id: worker_b,
                capability: Some(NodeCapability {
                    compute_score: 1,
                    available_ram_mb: 4096,
                }),
                wire_capability: None,
            },
            RouteWorkerCandidate {
                peer_id: worker_c,
                capability: Some(NodeCapability {
                    compute_score: 1,
                    available_ram_mb: 4096,
                }),
                wire_capability: None,
            },
            RouteWorkerCandidate {
                peer_id: worker_d,
                capability: Some(NodeCapability {
                    compute_score: 1,
                    available_ram_mb: 4096,
                }),
                wire_capability: None,
            },
        ];

        let route_plan = build_generation_route(
            17,
            Some(NodeCapability {
                compute_score: 1,
                available_ram_mb: 4096,
            }),
            &workers,
        )
        .expect("route planning should succeed");

        let routed_workers: Vec<_> = route_plan
            .worker_hops
            .iter()
            .map(|hop| hop.peer_id)
            .collect();
        assert_eq!(routed_workers, vec![worker_a, worker_b, worker_c, worker_d]);

        assert_eq!(route_plan.master_layer_range.start, 0);
        assert!(route_plan.master_layer_range.start < route_plan.master_layer_range.end);

        let mut cursor = route_plan.master_layer_range.end;
        for hop in &route_plan.worker_hops {
            assert_eq!(hop.layer_range.start, cursor);
            assert!(hop.layer_range.start < hop.layer_range.end);
            cursor = hop.layer_range.end;
        }
        assert_eq!(cursor, 17);
    }

    #[test]
    fn assigns_layer_range_per_worker_hop() {
        let worker_a = PeerId::random();
        let worker_b = PeerId::random();
        let route_plan = build_generation_route(
            8,
            Some(NodeCapability {
                compute_score: 1,
                available_ram_mb: 4096,
            }),
            &[
                RouteWorkerCandidate {
                    peer_id: worker_a,
                    capability: Some(NodeCapability {
                        compute_score: 1,
                        available_ram_mb: 4096,
                    }),
                    wire_capability: None,
                },
                RouteWorkerCandidate {
                    peer_id: worker_b,
                    capability: Some(NodeCapability {
                        compute_score: 1,
                        available_ram_mb: 4096,
                    }),
                    wire_capability: None,
                },
            ],
        )
        .expect("route planning should work");

        assert_eq!(route_plan.master_layer_range, 0..3);
        assert_eq!(route_plan.worker_hops[0].peer_id, worker_a);
        assert_eq!(route_plan.worker_hops[0].layer_range, 3..6);
        assert_eq!(route_plan.worker_hops[1].peer_id, worker_b);
        assert_eq!(route_plan.worker_hops[1].layer_range, 6..8);
    }

    #[test]
    fn route_planner_uses_weighted_capabilities_when_available() {
        let worker_a = PeerId::random();
        let worker_b = PeerId::random();
        let worker_c = PeerId::random();

        let workers = [
            RouteWorkerCandidate {
                peer_id: worker_a,
                capability: Some(NodeCapability {
                    compute_score: 3,
                    available_ram_mb: 100,
                }),
                wire_capability: None,
            },
            RouteWorkerCandidate {
                peer_id: worker_b,
                capability: Some(NodeCapability {
                    compute_score: 2,
                    available_ram_mb: 200,
                }),
                wire_capability: None,
            },
            RouteWorkerCandidate {
                peer_id: worker_c,
                capability: Some(NodeCapability {
                    compute_score: 1,
                    available_ram_mb: 300,
                }),
                wire_capability: None,
            },
        ];

        let first = build_generation_route(
            15,
            Some(NodeCapability {
                compute_score: 9,
                available_ram_mb: 500,
            }),
            &workers,
        )
        .expect("weighted route planning should work");
        let second = build_generation_route(
            15,
            Some(NodeCapability {
                compute_score: 9,
                available_ram_mb: 500,
            }),
            &workers,
        )
        .expect("weighted route planning should be deterministic");

        assert_eq!(first, second);
        assert_eq!(first.master_layer_range.start, 0);
        assert_eq!(
            first
                .worker_hops
                .iter()
                .map(|hop| hop.peer_id)
                .collect::<Vec<_>>(),
            vec![worker_a, worker_b, worker_c]
        );

        let total_layer_sum = (first.master_layer_range.end - first.master_layer_range.start)
            + first
                .worker_hops
                .iter()
                .map(|hop| hop.layer_range.end - hop.layer_range.start)
                .sum::<usize>();
        assert_eq!(total_layer_sum, 15);

        let mut cursor = first.master_layer_range.end;
        for hop in &first.worker_hops {
            assert_eq!(hop.layer_range.start, cursor);
            assert!(hop.layer_range.start < hop.layer_range.end);
            cursor = hop.layer_range.end;
        }
        assert_eq!(cursor, 15);
    }

    #[test]
    fn route_planner_rejects_unknown_memory_and_infeasible_routes() {
        let worker = RouteWorkerCandidate {
            peer_id: PeerId::random(),
            capability: None,
            wire_capability: None,
        };
        assert!(build_generation_route(8, None, &[]).is_err());
        assert!(
            build_generation_route(
                8,
                Some(NodeCapability {
                    compute_score: 1,
                    available_ram_mb: 4096
                }),
                &[worker]
            )
            .is_err()
        );
        assert!(
            build_generation_route(
                8,
                Some(NodeCapability {
                    compute_score: 1,
                    available_ram_mb: 50
                }),
                &[]
            )
            .is_err()
        );
    }

    #[test]
    fn wire_capability_never_shifts_workers_beyond_model_end() {
        let capability = crate::WorkerCapability {
            compute_score: 1,
            available_ram_mb: 4096,
            hardware: crate::HardwareSummary {
                has_gpu: false,
                backend: crate::ComputeBackend::Cpu,
                estimated_fp16_tflops: None,
                vram_total_bytes: None,
                vram_free_bytes: None,
                cpu_threads: 1,
            },
        };
        let worker = RouteWorkerCandidate {
            peer_id: PeerId::random(),
            capability: None,
            wire_capability: Some(capability),
        };
        let route = build_generation_route(
            12,
            Some(NodeCapability {
                compute_score: 1,
                available_ram_mb: 4096,
            }),
            &[worker],
        )
        .unwrap();
        assert_eq!(route.master_layer_range, 0..6);
        assert_eq!(route.worker_hops[0].layer_range, 6..12);
    }

    #[test]
    fn coordinator_can_own_no_transformer_layers() {
        let worker = RouteWorkerCandidate {
            peer_id: PeerId::random(),
            capability: Some(NodeCapability {
                compute_score: 1,
                available_ram_mb: 4096,
            }),
            wire_capability: None,
        };
        let route = build_generation_route(
            12,
            Some(NodeCapability {
                compute_score: 1,
                available_ram_mb: 0,
            }),
            &[worker],
        )
        .unwrap();
        assert_eq!(route.master_layer_range, 0..0);
        assert_eq!(route.worker_hops[0].layer_range, 0..12);
    }

    #[test]
    fn reports_failure_when_expected_worker_is_unavailable() {
        let worker_a = PeerId::random();
        let worker_b = PeerId::random();
        let route_plan = build_generation_route(
            8,
            Some(NodeCapability {
                compute_score: 1,
                available_ram_mb: 4096,
            }),
            &[
                RouteWorkerCandidate {
                    peer_id: worker_a,
                    capability: Some(NodeCapability {
                        compute_score: 1,
                        available_ram_mb: 4096,
                    }),
                    wire_capability: None,
                },
                RouteWorkerCandidate {
                    peer_id: worker_b,
                    capability: Some(NodeCapability {
                        compute_score: 1,
                        available_ram_mb: 4096,
                    }),
                    wire_capability: None,
                },
            ],
        )
        .expect("route planning should work");

        let available_workers: HashSet<_> = [worker_a].into_iter().collect();
        assert!(
            ensure_worker_available(route_plan.worker_hops[0].peer_id, &available_workers).is_ok()
        );

        let err = ensure_worker_available(route_plan.worker_hops[1].peer_id, &available_workers)
            .expect_err("missing worker should fail availability check");
        assert!(
            err.contains(&route_plan.worker_hops[1].peer_id.to_string()),
            "unexpected error: {err}"
        );
        assert!(err.contains("unavailable"), "unexpected error: {err}");
    }
}
