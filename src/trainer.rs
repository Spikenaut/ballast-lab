use crate::config::TrainingConfig;
use neuromod::{NeuroModulators, SpikingNetwork, StepError};
use thiserror::Error;

/// Summary of a training session.
#[derive(Debug, Default, Clone)]
pub struct TrainingSummary {
    pub steps_processed: usize,
    pub total_spikes: u64,
    pub avg_reward: f32,
    pub threshold_drifts: Vec<f32>,
    pub weight_drifts: Vec<Vec<f32>>,
    pub per_neuron_spikes: Vec<u64>,
}

/// Input sample for a generic training session.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub stimuli: Vec<f32>,
    pub reward: f32,
}

#[derive(Debug, Error)]
pub enum TrainerError {
    #[error("network step failed: {0:?}")]
    Step(StepError),
    #[error("empty training batch")]
    EmptyBatch,
}

/// The SpikenautTrainer manages the evolution of SNN parameters using reward-modulated STDP.
pub struct SpikenautTrainer {
    pub config: TrainingConfig,
}

impl SpikenautTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Runs a training step using generic stimuli and an externally computed reward.
    pub fn train_step(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        reward: f32,
    ) -> Result<Vec<usize>, StepError> {
        let mut modulators: NeuroModulators = network.modulators;

        // Positive reward shifts toward dopamine; negative toward cortisol.
        if reward > 0.0 {
            modulators.dopamine = (modulators.dopamine + reward * 0.1).clamp(0.0, 1.0);
            modulators.cortisol = (modulators.cortisol - reward * 0.05).clamp(0.0, 1.0);
        } else {
            modulators.cortisol = (modulators.cortisol - reward * 0.2).clamp(0.0, 1.0);
            modulators.dopamine = (modulators.dopamine + reward * 0.1).clamp(0.0, 1.0);
        }

        network.step(stimuli, &modulators)
    }

    /// Replays a batch of generic training examples.
    pub fn run_session(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        if data.is_empty() {
            return Err(TrainerError::EmptyBatch);
        }

        let mut summary = TrainingSummary::default();
        let initial_thresholds = network.get_thresholds();
        let initial_weights: Vec<Vec<f32>> =
            network.neurons.iter().map(|n| n.weights.clone()).collect();

        summary.per_neuron_spikes = vec![0; network.neurons.len()];
        let mut total_reward = 0.0;

        for example in data {
            let spikes = self
                .train_step(network, &example.stimuli, example.reward)
                .map_err(TrainerError::Step)?;
            total_reward += example.reward;
            summary.steps_processed += 1;

            summary.total_spikes += spikes.len() as u64;
            for &idx in &spikes {
                if idx < summary.per_neuron_spikes.len() {
                    summary.per_neuron_spikes[idx] += 1;
                }
            }
        }

        summary.avg_reward = total_reward / data.len() as f32;

        let final_thresholds = network.get_thresholds();
        for i in 0..network.neurons.len() {
            summary
                .threshold_drifts
                .push(final_thresholds[i] - initial_thresholds[i]);

            let mut w_deltas = Vec::new();
            for (ch, &w) in network.neurons[i].weights.iter().enumerate() {
                w_deltas.push(w - initial_weights[i][ch]);
            }
            summary.weight_drifts.push(w_deltas);
        }

        Ok(summary)
    }
}
