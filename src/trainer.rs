use neuromod::SpikingNetwork;
use spikenaut_telemetry::TelemetrySnapshot;
use spikenaut_reward::MiningRewardState;
use crate::config::TrainingConfig;
use std::collections::VecDeque;

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

/// The SpikenautTrainer manages the evolution of SNN parameters using reward-modulated STDP.
pub struct SpikenautTrainer {
    pub config: TrainingConfig,
    reward_state: MiningRewardState,
    history: VecDeque<TelemetrySnapshot>,
}

impl SpikenautTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            reward_state: MiningRewardState::new(),
            history: VecDeque::with_capacity(1000),
        }
    }

    /// Runs a training step using a single telemetry snapshot.
    /// 
    /// This mimics the "closed loop" training:
    /// 1. Ingest telemetry.
    /// 2. Compute reward (dopamine/cortisol).
    /// 3. Update SNN modulators.
    /// 4. Step SNN (integration + spikes + STDP).
    pub fn train_step(
        &mut self,
        network: &mut SpikingNetwork,
        snapshot: &TelemetrySnapshot,
    ) -> f32 {
        // 1. Compute Reward (Mining Dopamine)
        let gpu_telemetry = spikenaut_reward::GpuTelemetry {
            hashrate_mh: snapshot.dynex_hashrate_mh as f32,
            power_w: snapshot.gpu_power_w,
            gpu_temp_c: snapshot.gpu_temp_c,
            gpu_clock_mhz: snapshot.gpu_clock_mhz,
            vddcr_gfx_v: snapshot.gpu_voltage_v,
            ..Default::default()
        };
        
        let reward = self.reward_state.compute(&gpu_telemetry, Some(snapshot.cpu_tctl_c));

        // 2. Map reward to neuromodulators
        // Positive reward -> Dopamine boost
        // Negative reward -> Cortisol boost
        if reward > 0.0 {
            network.modulators.dopamine = (network.modulators.dopamine + reward * 0.1).clamp(0.0, 1.0);
            network.modulators.cortisol = (network.modulators.cortisol - reward * 0.05).clamp(0.0, 1.0);
        } else {
            network.modulators.cortisol = (network.modulators.cortisol - reward * 0.2).clamp(0.0, 1.0); // reward is negative
            network.modulators.dopamine = (network.modulators.dopamine + reward * 0.1).clamp(0.0, 1.0);
        }

        // 3. Encode telemetry to spikes (simplified sum for now, matches spikenaut-hybrid logic)
        let mut stimuli = [0.0f32; neuromod::NUM_INPUT_CHANNELS];
        stimuli[0] = (snapshot.gpu_voltage_v - 1.0).abs() * 2.0;
        stimuli[1] = snapshot.gpu_power_w / 450.0;
        stimuli[2] = snapshot.dynex_hashrate_mh as f32 / 0.015;
        stimuli[3] = (snapshot.gpu_temp_c - 50.0) / 40.0;
        
        // 4. Step the network (includes STDP internally in neuromod::engine)
        network.step(&stimuli, &network.modulators.clone());

        reward
    }

    /// Replays a batch of telemetry data for training.
    pub fn run_session(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TelemetrySnapshot],
    ) -> TrainingSummary {
        let mut summary = TrainingSummary::default();
        let initial_thresholds = network.get_thresholds();
        let initial_weights: Vec<Vec<f32>> = network.neurons.iter().map(|n| n.weights.clone()).collect();
        
        summary.per_neuron_spikes = vec![0; network.neurons.len()];
        let mut total_reward = 0.0;

        for snapshot in data {
            let reward = self.train_step(network, snapshot);
            total_reward += reward;
            summary.steps_processed += 1;

            for (i, neuron) in network.neurons.iter().enumerate() {
                if neuron.last_spike {
                    summary.total_spikes += 1;
                    summary.per_neuron_spikes[i] += 1;
                }
            }
        }

        summary.avg_reward = if !data.is_empty() { total_reward / data.len() as f32 } else { 0.0 };

        // Calculate deltas
        let final_thresholds = network.get_thresholds();
        for i in 0..network.neurons.iter().len() {
            summary.threshold_drifts.push(final_thresholds[i] - initial_thresholds[i]);
            
            let mut w_deltas = Vec::new();
            for (ch, &w) in network.neurons[i].weights.iter().enumerate() {
                w_deltas.push(w - initial_weights[i][ch]);
            }
            summary.weight_drifts.push(w_deltas);
        }

        summary
    }
}
