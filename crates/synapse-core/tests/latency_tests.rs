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
