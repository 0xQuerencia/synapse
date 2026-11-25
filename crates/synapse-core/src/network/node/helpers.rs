use libp2p::{PeerId, request_response};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::{ChatEvent, StopReason, SynapseNode, SynapseRole, WorkerCapabilitySnapshot};
use crate::{ModelIdentity, NodeHello, NodeRole};

impl SynapseNode {
    /// Send a ChatEvent to the CLI (no-op if channels aren't configured).
    pub(super) fn emit(&mut self, event: ChatEvent) {
        if let Some(tx) = &self.event_tx
            && let Err(error) = tx.try_send(event)
        {
            warn!(%error, "requester cannot accept output; cancelling generation");
            self.event_tx = None;
            self.generation_active = false;
            self.local_step_pending = false;
            self.release_generation_assignments();
        }
    }

    pub(super) fn fail_generation(&mut self, message: impl Into<String>) {
        self.release_generation_assignments();
        let message = message.into();
        warn!(%message, "generation failed");
        self.emit(ChatEvent::Token(format!("[generation-error] {message}")));
        if self.generation_active {
            self.emit(ChatEvent::GenerationDone {
                tokens_generated: self.tokens_generated,
                reason: StopReason::Error,
            });
        }
        self.generation_active = false;
        self.awaiting_clear_cache_ack = false;
        self.pending_cache_reset_trace_id = None;
        self.pending_cache_reset_peers.clear();
        self.generation_route_plan = None;
        self.in_flight_activation_hop = None;
        // Reset healing state
        self.healing_in_progress = false;
        self.healing_saved_tokens = None;
        // Reset batch state
        self.batch_active_max_tokens = None;
        // Reset speculative state
        self.spec_round = None;
        self.awaiting_truncate_ack = false;
        self.pending_truncate_trace_id = None;
        self.pending_truncate_peers.clear();
    }

    pub(super) fn send_worker_hello(&mut self, peer: PeerId) {
        let SynapseRole::Worker { .. } = &self.role else {
            return;
        };

        let explicitly_selected = matches!(&self.role, SynapseRole::Worker { manual_master: Some(addr), .. } if extract_peer_id(addr) == Some(peer));
        if !explicitly_selected
            && !self
                .net_config
                .discovery
                .allowed_peers
                .contains(&peer.to_string())
        {
            return;
        }
        if !self.greeted_peers.insert(peer) {
            debug!(%peer, "worker already sent NodeHello to peer");
            return;
        }

        let mut hello = node_hello(NodeRole::Worker, self.model.clone());
        // If the GUI ran a real compute benchmark, advertise that measured
        // score instead of the coarse probe heuristic (cpu_threads × 100).
        // The master reads `hello.compute_score` and the weighted planner
        // divides layers proportionally by it, so this is what actually
        // makes "divide power accordingly" use real measurements.
        if let Some(score) = self.measured_compute_score {
            hello.compute_score = Some(score);
        }
        hello.available_ram_mb = Some(self.memory_limit_bytes.div_ceil(1024 * 1024));
        hello.memory_bytes = Some(self.memory_limit_bytes);
        if let Some(capability) = hello.capability.as_mut() {
            capability.compute_score = hello.compute_score.unwrap_or(1);
            capability.available_ram_mb = hello.available_ram_mb.unwrap_or(0);
        }
        info!(%peer, "worker sending NodeHello");
        if let Err(error) = self.send_tracked_request(
            peer,
            super::super::RequestMessage::NodeHello {
                hello,
                pairing_code: String::new(),
            },
        ) {
            warn!(%peer, %error, "failed to start pairing");
            self.greeted_peers.remove(&peer);
        }
    }

    pub(super) fn reject(
        &mut self,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
        reason: &str,
    ) {
        self.send_response(channel, super::super::ResponseMessage::rejected(reason));
    }

    pub(super) fn send_response(
        &mut self,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
        response: super::super::ResponseMessage,
    ) {
        if let Err(_response) = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_response(channel, response)
        {
            error!("response channel unavailable");
        }
    }

    /// Send a HealthPing to pre-warm the connection with a worker.
    /// Called immediately after authentication to eliminate cold-start latency.
    pub(super) fn send_health_ping(&mut self, peer: PeerId) {
        info!(%peer, "sending HealthPing to pre-warm connection");
        if let Err(error) =
            self.send_tracked_request(peer, super::super::RequestMessage::HealthPing)
        {
            warn!(%peer, %error, "health request not sent");
        }
    }
}

/// Helper for tokio::select! — returns pending if no batch queue is configured.
pub(super) async fn recv_prompt(rx: &mut Option<mpsc::Receiver<String>>) -> Option<String> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

pub(super) async fn recv_batch_prompt(
    rx: &mut Option<mpsc::UnboundedReceiver<super::BatchPromptRequest>>,
) -> Option<super::BatchPromptRequest> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

pub(super) fn local_worker_capability_snapshot() -> Option<WorkerCapabilitySnapshot> {
    let cpu_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1)
        .min(u16::MAX as usize) as u16;
    let compute_score = Some(u64::from(cpu_threads).saturating_mul(100));
    let available_ram_mb = detect_available_ram_mb();

    let snapshot = WorkerCapabilitySnapshot {
        compute_score,
        available_ram_mb,
        available_memory_bytes: available_ram_mb.and_then(|mb| mb.checked_mul(1024 * 1024)),
    };
    if snapshot.compute_score.is_none() && snapshot.available_ram_mb.is_none() {
        return None;
    }
    Some(snapshot)
}

fn node_hello(role: NodeRole, model: ModelIdentity) -> NodeHello {
    let probe = crate::hardware::HardwareProbe::detect();
    NodeHello::from_probe(role, model, std::env::consts::OS, &probe)
}

/// Detect available system RAM in MiB. Platform-specific implementation.
#[cfg(target_os = "linux")]
/// Detect available system RAM in MiB. Platform-specific implementation.
/// Public so the compute-score benchmark path can read it.
#[cfg(target_os = "linux")]
pub(crate) fn detect_available_ram_mb() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut available_kib = None;

    for line in meminfo.lines() {
        if available_kib.is_none() && line.starts_with("MemAvailable:") {
            available_kib = parse_meminfo_kib(line);
        }
    }

    // Total memory is not usable headroom. Respect a cgroup's remaining
    // memory where this process is containerized.
    let selected_kib = available_kib?;
    let host_bytes = selected_kib.checked_mul(1024)?;
    let cgroup_headroom = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
        .ok()
        .and_then(|limit| limit.trim().parse::<u64>().ok())
        .and_then(|limit| {
            std::fs::read_to_string("/sys/fs/cgroup/memory.current")
                .ok()
                .and_then(|used| used.trim().parse::<u64>().ok())
                .map(|used| limit.saturating_sub(used))
        });
    let mib = cgroup_headroom.map_or(host_bytes, |bytes| bytes.min(host_bytes)) / (1024 * 1024);
    (mib > 0).then_some(mib)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse::<u64>().ok()
}

/// Detect available system RAM in MiB via `sysctl hw.memsize`.
#[cfg(target_os = "macos")]
pub(crate) fn detect_available_ram_mb() -> Option<u64> {
    use std::process::Command;
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let bytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    let mib = bytes / (1024 * 1024);
    (mib > 0).then_some(mib)
}

/// Detect available system RAM in MiB via PowerShell.
#[cfg(target_os = "windows")]
pub(crate) fn detect_available_ram_mb() -> Option<u64> {
    use std::process::Command;
    let output = Command::new("powershell")
        .args([
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
        ])
        .output()
        .ok()?;
    let kib: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    let mib = kib / 1024;
    (mib > 0).then_some(mib)
}

/// Fallback for unsupported platforms — logs a warning and returns None.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn detect_available_ram_mb() -> Option<u64> {
    warn!("RAM detection not supported on this platform");
    None
}

pub(super) fn extract_peer_id(addr: &libp2p::Multiaddr) -> Option<PeerId> {
    for p in addr.iter() {
        if let libp2p::multiaddr::Protocol::P2p(peer_id) = p {
            return Some(peer_id);
        }
    }
    None
}

/// Queue approved relay connections. Enqueue time is not network RTT.
pub(super) fn queue_relay_connections(
    swarm: &mut libp2p::Swarm<super::SynapseBehaviour>,
    relays: &[libp2p::Multiaddr],
) -> Vec<crate::RelayInfo> {
    relays
        .iter()
        .map(|addr| {
            if let Err(error) = swarm.dial(addr.clone()) {
                warn!(%addr, %error, "failed to queue relay connection");
            }
            crate::RelayInfo::new(addr.clone())
        })
        .collect()
}
