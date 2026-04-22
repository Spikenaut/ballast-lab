pub mod config;
pub mod trainer;

pub use config::TrainingConfig;
pub use trainer::{SpikenautTrainer, TrainerError, TrainingExample, TrainingSummary};
