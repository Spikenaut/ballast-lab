pub mod config;
pub mod trainer;

pub use config::TrainingConfig;
pub use trainer::{SpikenautTrainer, TrainingSummary};

/// Trait for types that can provide telemetry for training.
pub trait TelemetrySource {
    fn next_snapshot(&mut self) -> Option<spikenaut_telemetry::TelemetrySnapshot>;
}

/// Trait for custom reward functions.
pub trait RewardFunction {
    fn compute_reward(
        &self,
        snapshot: &spikenaut_telemetry::TelemetrySnapshot,
    ) -> f32;
}

/// A default reward function using spikenaut-reward's MiningRewardState logic.
pub struct DefaultMiningReward {
    pub state: std::sync::Mutex<spikenaut_reward::MiningRewardState>,
}

impl Default for DefaultMiningReward {
    fn default() -> Self {
        Self {
            state: std::sync::Mutex::new(spikenaut_reward::MiningRewardState::new()),
        }
    }
}

impl RewardFunction for DefaultMiningReward {
    fn compute_reward(
        &self,
        snapshot: &spikenaut_telemetry::TelemetrySnapshot,
    ) -> f32 {
        let mut state = self.state.lock().unwrap();
        let gpu_telemetry = spikenaut_reward::GpuTelemetry {
            hashrate_mh: snapshot.dynex_hashrate_mh as f32,
            power_w: snapshot.gpu_power_w,
            gpu_temp_c: snapshot.gpu_temp_c,
            gpu_clock_mhz: snapshot.gpu_clock_mhz,
            vddcr_gfx_v: snapshot.gpu_voltage_v,
            ..Default::default()
        };
        state.compute(&gpu_telemetry, Some(snapshot.cpu_tctl_c))
    }
}
