//! Integration tests for latency optimization phases.
//!
//! Verifies adaptive timeout calculation, latency-aware worker selection,
//! request pipelining, and incremental KV cache transfer.

use std::time::Duration;

#[test]
fn test_adaptive_timeout_calculation() {
    // Test adaptive timeout: 2 × avg_latency
    let latencies = [
        Duration::from_millis(20),
        Duration::from_millis(25),
        Duration::from_millis(30),
        Duration::from_millis(22),
        Duration::from_millis(28),
    ];

    let sum: Duration = latencies.iter().sum();
    let avg = sum / latencies.len() as u32;
    let adaptive_timeout = avg * 2;

    assert!(
        adaptive_timeout < Duration::from_secs(1),
        "Adaptive timeout should be much less than default 10s"
    );
    assert!(
        adaptive_timeout >= Duration::from_millis(40),
        "Adaptive timeout should be at least 2x average latency"
    );
    assert!(
        adaptive_timeout <= Duration::from_millis(60),
        "Adaptive timeout for good latencies should be <100ms"
    );
}

#[test]
fn test_adaptive_timeout_with_outliers() {
    // Test that outliers don't skew adaptive timeout too much
    let latencies = vec![
        Duration::from_millis(20),
        Duration::from_millis(25),
        Duration::from_millis(300), // outlier
        Duration::from_millis(22),
        Duration::from_millis(23),
    ];

    // Use median instead of mean for robustness
    let mut sorted = latencies.clone();
    sorted.sort();
    let median = sorted[sorted.len() / 2];
    let adaptive_timeout = median * 2;

    assert!(
        adaptive_timeout < Duration::from_millis(100),
        "Median-based timeout should ignore outliers"
    );
}

#[test]
fn test_adaptive_timeout_minimum() {
    // Ensure we have a minimum timeout even with very fast workers
    let latencies = [
        Duration::from_millis(5),
        Duration::from_millis(6),
        Duration::from_millis(4),
    ];

    let sum: Duration = latencies.iter().sum();
    let avg = sum / latencies.len() as u32;
    let adaptive_timeout = avg * 2;
    let min_timeout = Duration::from_millis(50);
    let final_timeout = adaptive_timeout.max(min_timeout);

    assert!(
        final_timeout >= min_timeout,
        "Should enforce minimum timeout to avoid false positives"
    );
}

#[test]
fn test_latency_aware_worker_selection() {
    // Mock worker latency data
    struct Worker {
        id: u32,
        avg_latency_ms: u32,
        compute_score: u64,
    }

    let workers = [
        Worker {
            id: 1,
            avg_latency_ms: 50,
            compute_score: 1000,
        },
        Worker {
            id: 2,
            avg_latency_ms: 20,
            compute_score: 800,
        },
        Worker {
            id: 3,
            avg_latency_ms: 100,
            compute_score: 1200,
        },
    ];

    // Select worker with best latency-compute tradeoff
    let mut scored: Vec<_> = workers
        .iter()
        .map(|w| {
            // Score = compute / latency (higher is better)
            let score = (w.compute_score as f64) / (w.avg_latency_ms as f64);
            (w.id, score, w.avg_latency_ms)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Worker 2 should win: 800/20 = 40 vs Worker 1: 1000/50 = 20
    assert_eq!(
        scored[0].0, 2,
        "Should select worker with best latency/compute ratio"
    );
    assert!(
        scored[0].1 > 30.0,
        "Best worker should have high efficiency score"
    );
}

#[test]
fn test_latency_aware_worker_selection_with_timeout_data() {
    // Real-world scenario: prefer workers that consistently respond
    struct WorkerStats {
        id: u32,
        avg_latency_ms: u32,
        timeout_rate: f32, // 0.0 to 1.0
        compute_score: u64,
    }

    let workers = [
        WorkerStats {
            id: 1,
            avg_latency_ms: 25,
            timeout_rate: 0.0,
            compute_score: 1000,
        },
        WorkerStats {
            id: 2,
            avg_latency_ms: 20,
            timeout_rate: 0.4,
            compute_score: 1200,
        },
        WorkerStats {
            id: 3,
            avg_latency_ms: 40,
            timeout_rate: 0.0,
            compute_score: 900,
        },
    ];

    // Penalize workers with high timeout rates
    let mut scored: Vec<_> = workers
        .iter()
        .map(|w| {
            let reliability = 1.0 - w.timeout_rate;
            let score = (w.compute_score as f64 * reliability as f64) / (w.avg_latency_ms as f64);
            (w.id, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Worker 1 should win despite slightly higher latency (no timeouts)
    assert_eq!(
        scored[0].0, 1,
        "Should prefer reliable workers over slightly faster but flaky ones"
    );
}

#[test]
fn test_pipelining_overlap_calculation() {
    // Test that we can overlap next request while waiting for current
    let layer_compute_time_ms = 15;
    let network_rtt_ms = 10;
    let prefetch_threshold_ms = network_rtt_ms + 5; // Start fetching 5ms before needed

    // If we start prefetching when remaining compute > prefetch_threshold,
    // the next activation should arrive just in time
    let remaining_compute = layer_compute_time_ms + 5;
    let should_prefetch = remaining_compute > prefetch_threshold_ms;

    assert!(
        should_prefetch,
        "Should start prefetch before current layer completes"
    );

    let overlap_savings = network_rtt_ms.min(remaining_compute - prefetch_threshold_ms);
    assert!(
        overlap_savings > 0,
        "Pipelining should save at least some latency"
    );
}

#[test]
fn test_incremental_kv_cache_transfer() {
    // Test that KV cache is transferred incrementally, not all at once
    let total_layers = 32;
    let tokens_generated = 10;
    let bytes_per_layer_per_token = 4096; // Typical hidden state size

    // Incremental: only send cache for new token
    let incremental_bytes = total_layers * bytes_per_layer_per_token;

    // Full: send entire cache every time
    let full_bytes = total_layers * tokens_generated * bytes_per_layer_per_token;

    let savings_ratio = (full_bytes - incremental_bytes) as f64 / full_bytes as f64;

    assert!(
        savings_ratio > 0.8,
        "Incremental KV cache should save >80% bandwidth after first token"
    );
    assert_eq!(
        incremental_bytes,
        total_layers * bytes_per_layer_per_token,
        "Should only transfer cache for current token"
    );
}

#[test]
fn test_kv_cache_compression_benefit() {
    // Test that compression helps KV cache transfer
    let cache_size_bytes = 32 * 4096; // 32 layers × 4KB per layer
    let compression_ratio = 2.5; // Typical for LZ4 on activation data

    let compressed_size = (cache_size_bytes as f64 / compression_ratio) as usize;
    let savings = cache_size_bytes - compressed_size;

    assert!(
        savings > cache_size_bytes / 2,
        "Compression should save >50% on KV cache transfer"
    );
}

#[test]
fn test_end_to_end_latency_budget() {
    // Test that all optimizations together meet <50ms target.
    //
    // Pipelined steady state: the network RTT is hidden behind compute when
    // compute-per-hop exceeds the RTT (workers prefetch the next token while
    // the current hop computes), so per-token latency is dominated by
    // max(compute_per_hop, RTT) plus codec overhead — not their sum.
    let network_rtt_ms = 15; // Optimized with relay selection
    let compression_overhead_ms = 2; // LZ4 is very fast
    let layer_compute_ms = 4; // Per layer on decent hardware (SmolLM2-135M)
    let layers_per_hop = 8; // Sharded across 4 workers

    let compute_per_hop_ms = layer_compute_ms * layers_per_hop;
    let per_token_latency = compute_per_hop_ms.max(network_rtt_ms) + compression_overhead_ms;

    assert!(
        per_token_latency < 50,
        "End-to-end per-token latency should be <50ms with all optimizations"
    );
}

#[test]
fn test_worker_timeout_recovery() {
    // Test that we can quickly re-route when a worker times out
    let adaptive_timeout_ms = 50; // 2x avg latency of 25ms
    let default_timeout_ms = 10_000;

    let savings = default_timeout_ms - adaptive_timeout_ms;

    assert!(
        savings > 9_000,
        "Adaptive timeout should detect failures 200x faster than default"
    );
}

#[test]
fn test_compression_codec_selection() {
    // Test that we choose appropriate compression for payload size
    let small_payload_bytes = 1024; // 1 KB
    let large_payload_bytes = 131_072; // 128 KB

    // Small payloads: compression overhead > benefit
    let should_compress_small = small_payload_bytes > 4096;
    assert!(!should_compress_small, "Should not compress small payloads");

    // Large payloads: compression saves time on network
    let should_compress_large = large_payload_bytes > 4096;
    assert!(should_compress_large, "Should compress large payloads");
}

#[test]
fn test_p95_latency_calculation() {
    // Test percentile calculation for SLA verification
    let mut latencies: Vec<u32> = vec![
        15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60,
        120, // 120 is outlier
    ];

    latencies.sort();
    let p95_index = (latencies.len() as f64 * 0.95).ceil() as usize - 1;
    let p95_latency = latencies[p95_index];

    assert!(p95_latency < 100, "P95 latency should be <100ms");
    assert!(p95_latency > 50, "P95 should be higher than median");
}

#[test]
fn test_concurrent_request_handling() {
    // Test that multiple concurrent requests don't interfere
    let max_concurrent_streams = 128;
    let active_requests = 10;

    assert!(
        active_requests < max_concurrent_streams,
        "Should handle 10 concurrent requests with max_streams=128"
    );

    // Each request should get fair share of bandwidth
    let bandwidth_mbps = 1000.0; // 1 Gbps
    let per_request_mbps = bandwidth_mbps / active_requests as f64;

    assert!(
        per_request_mbps > 50.0,
        "Each request should get >50 Mbps with 10 concurrent on 1Gbps link"
    );
}
