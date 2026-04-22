# plasticity-lab

Generic reward-modulated plasticity loops for spiking neural networks.

## Overview

`plasticity-lab` provides a small, reusable training loop around `neuromod::SpikingNetwork`.
It is intentionally domain-agnostic:

- Input encoding belongs to `axon-encoder`
- Reward shaping belongs to `limbic-critic`
- This crate runs the loop and tracks training summaries

## Usage

```rust
use neuromod::SpikingNetwork;
use plasticity_lab::{SpikenautTrainer, TrainingConfig, TrainingExample};

let mut trainer = SpikenautTrainer::new(TrainingConfig::default());
let mut network = SpikingNetwork::with_dimensions(32, 8, 64);

let batch = vec![
    TrainingExample {
        stimuli: vec![0.25; 64],
        reward: 0.2,
    },
    TrainingExample {
        stimuli: vec![0.4; 64],
        reward: -0.1,
    },
];

let summary = trainer.run_session(&mut network, &batch).unwrap();
println!("processed={}, avg_reward={}", summary.steps_processed, summary.avg_reward);
```

## Notes

- `train_step` accepts dynamic `&[f32]` stimuli and returns `Result<Vec<usize>, StepError>`.
- `run_session` returns a `TrainingSummary` with spike counts and parameter drift metrics.
- `axon-encoder` and `limbic-critic` are optional integration dependencies under the `integration` feature.

## License

GPL-3.0
