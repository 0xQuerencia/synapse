use libp2p::PeerId;
use std::time::Instant;

/// Batch size calibration state for a single worker.
#[derive(Clone, Debug)]
pub struct BatchCalibration {
    pub peer_id: PeerId,
    /// Batch sizes to test: [1, 4, 8, 16, 32].
    pub test_sizes: Vec<usize>,
    /// Current test index in `test_sizes`.
    pub current_test_index: usize,
    /// Measured throughput (tokens/sec) for each test size.
    pub measured_throughput: Vec<f32>,
    /// Number of test samples collected for the current batch size.
    pub samples_collected: usize,
    /// Sum of throughput samples for the current batch size (used for averaging).
    pub sample_sum: f32,
    /// Target number of samples per batch size.
    pub samples_per_size: usize,
}

impl BatchCalibration {
    /// Start a new calibration for a worker.
    pub fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            test_sizes: vec![1, 4, 8, 16, 32],
            current_test_index: 0,
            measured_throughput: Vec::new(),
            samples_collected: 0,
            sample_sum: 0.0,
            samples_per_size: 5, // 5 samples per batch size
        }
    }

    /// Get the current test batch size.
