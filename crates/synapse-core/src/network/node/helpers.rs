use libp2p::{PeerId, request_response};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::{ChatEvent, StopReason, SynapseNode, SynapseRole, WorkerCapabilitySnapshot};
use crate::{ModelIdentity, NodeHello, NodeRole};

impl SynapseNode {
    /// Send a ChatEvent to the CLI (no-op if channels aren't configured).
    pub(super) fn emit(&self, event: ChatEvent) {
        if let Some(tx) = &self.event_tx
            && let Err(error) = tx.try_send(event)
        {
            warn!(%error, "dropping chat event because channel is full or closed");
        }
    }

    pub(super) fn fail_generation(&mut self, message: impl Into<String>) {
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
        let SynapseRole::Worker { pairing_code, .. } = &self.role else {
            return;
        };

        if !self.greeted_peers.insert(peer) {
            debug!(%peer, "worker already sent NodeHello to peer");
            return;
        }

        let hello = node_hello(NodeRole::Worker, self.model.clone());
        let hashed_code = crate::hash_pairing_code(pairing_code);
        info!(%peer, "worker sending NodeHello");
        self.swarm.behaviour_mut().request_response.send_request(
            &peer,
            super::super::RequestMessage::NodeHello {
                hello,
                pairing_code: hashed_code,
            },
        );
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
        if let Err(response) = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_response(channel, response)
        {
            error!(?response, "failed to send response");
        }
    }
}

/// Helper for tokio::select! — returns pending if no channel is configured.
pub(super) async fn recv_prompt(rx: &mut Option<mpsc::Receiver<String>>) -> Option<String> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// Helper for tokio::select! — returns pending if no batch queue is configured.
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
    };
    if snapshot.compute_score.is_none() && snapshot.available_ram_mb.is_none() {
        return None;
    }
    Some(snapshot)
}

fn node_hello(role: NodeRole, model: ModelIdentity) -> NodeHello {
    let cpu_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1)
        .min(u16::MAX as usize) as u16;
    let capability = local_worker_capability_snapshot();
    let compute_score = capability.as_ref().and_then(|entry| entry.compute_score);
    let available_ram_mb = capability.as_ref().and_then(|entry| entry.available_ram_mb);
    let memory_bytes = available_ram_mb.and_then(|mb| mb.checked_mul(1024 * 1024));

    NodeHello {
        protocol_version: PROTOCOL_VERSION.to_owned(),
        role,
        model,
        platform: std::env::consts::OS.to_owned(),
        cpu_threads,
        memory_bytes,
        compute_score,
        available_ram_mb,
    }
}

/// Detect available system RAM in MiB. Platform-specific implementation.
#[cfg(target_os = "linux")]
fn detect_available_ram_mb() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kib = None;
    let mut available_kib = None;

    for line in meminfo.lines() {
        if available_kib.is_none() && line.starts_with("MemAvailable:") {
            available_kib = parse_meminfo_kib(line);
        } else if total_kib.is_none() && line.starts_with("MemTotal:") {
            total_kib = parse_meminfo_kib(line);
        }
    }

    let selected_kib = available_kib.or(total_kib)?;
    let mib = selected_kib / 1024;
    (mib > 0).then_some(mib)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse::<u64>().ok()
}

/// Detect available system RAM in MiB via `sysctl hw.memsize`.
#[cfg(target_os = "macos")]
fn detect_available_ram_mb() -> Option<u64> {
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
fn detect_available_ram_mb() -> Option<u64> {
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
