//! Adaptive timeout calculation for Phase 1 Task 1.5.
//!
//! Dynamically adjusts worker hop timeouts based on observed latency patterns
//! to enable faster failure detection and re-routing.

use std::time::Duration;

use super::{SynapseNode, WORKER_HOP_TIMEOUT};

impl SynapseNode {
    /// Compute adaptive timeout for a given hop index.
    ///
    /// Strategy: Use 2 × moving average of recent hop latencies for this layer.
    /// Falls back to per-worker latency if no hop-specific data exists.
    /// Falls back to static `WORKER_HOP_TIMEOUT` if no historical data.
    ///
    /// This allows faster timeout detection (e.g., 2-4s instead of 10s) when
    /// workers are consistently fast, while still tolerating slow workers.
    pub(super) fn compute_adaptive_timeout(&self, hop_index: usize) -> Duration {
        // Try hop-specific latency first (most accurate for layer characteristics)
        if let Some(latencies) = self.hop_latencies.get(&hop_index)
            && !latencies.is_empty()
        {
            let sum: Duration = latencies.iter().sum();
            let avg = sum / latencies.len() as u32;
            // 2× safety margin
            return avg * 2;
        }

        // Fall back to the in-flight worker's historical latency
        if let Some(_hop) = &self.in_flight_activation_hop
            && let Some(route_plan) = &self.generation_route_plan
            && let Some(worker_hop) = route_plan.worker_hops.get(hop_index)
        {
            let peer = worker_hop.peer_id;
            if let Some(latencies) = self.worker_latencies.get(&peer)
                && !latencies.is_empty()
            {
                let sum: Duration = latencies.iter().sum();
                let avg = sum / latencies.len() as u32;
                return avg * 2;
            }
        }

        // No historical data — use static timeout
        WORKER_HOP_TIMEOUT
    }
}
