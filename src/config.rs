use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub target_spikes_per_step: f32,
    pub homeostasis_strength: f32,
    pub batch_size: usize,
    pub use_reward_modulation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            target_spikes_per_step: 0.1,
            homeostasis_strength: 0.001,
            batch_size: 1,
            use_reward_modulation: true,
        }
    }
}
