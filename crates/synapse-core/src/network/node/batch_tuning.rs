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
    pub fn current_batch_size(&self) -> usize {
        self.test_sizes
            .get(self.current_test_index)
            .copied()
            .unwrap_or(8)
    }

    /// Record a throughput measurement for the current batch size.
    ///
    /// Returns `true` if calibration is complete.
    pub fn record_sample(&mut self, throughput: f32) -> bool {
        self.sample_sum += throughput;
        self.samples_collected += 1;

        if self.samples_collected >= self.samples_per_size {
            // Average only this batch size's samples (not a rolling mean
            // across previously recorded batch sizes).
            let avg = self.sample_sum / self.samples_per_size as f32;
            self.measured_throughput.push(avg);
            self.sample_sum = 0.0;
            self.samples_collected = 0;
            self.current_test_index += 1;

            // Check if we've tested all batch sizes
            self.current_test_index >= self.test_sizes.len()
        } else {
            false
        }
    }

    /// Get the optimal batch size and throughput after calibration completes.
    pub fn optimal_result(&self) -> Option<(usize, f32)> {
        if self.measured_throughput.len() != self.test_sizes.len() {
            return None;
        }

        let mut best_idx = 0;
