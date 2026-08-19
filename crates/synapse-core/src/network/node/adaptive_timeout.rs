//! Adaptive timeout calculation for Phase 1 Task 1.5.
//!
//! Dynamically adjusts worker hop timeouts based on observed latency patterns
//! to enable faster failure detection and re-routing.

use std::time::Duration;

use super::{SynapseNode, WORKER_HOP_TIMEOUT};

impl SynapseNode {
    /// Compute adaptive timeout for a given hop index.
    ///
