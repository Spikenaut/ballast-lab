# ballast-lab

Reward-modulated training loops for Spikenaut SNN/LLM fusion.

## Overview

This crate provides the training infrastructure for neuromorphic networks in the Spikenaut ecosystem, extracted from the proven crypto training loops of the original `ship_of_theseus_rs` repository.

### Features

- **Reward-Modulated STDP**: Spike-timing-dependent plasticity with dopamine/cortisol modulation
- **Telemetry-to-Spike Encoding**: Converts hardware telemetry into spike trains
- **Mining Efficiency Rewards**: Multi-dimensional reward signal from GPU/CPU telemetry
- **Closed-Loop Training**: Complete telemetry → encoding → neuromodulation → weight update pipeline
- **Clean Trait Boundaries**: Pluggable telemetry sources and reward functions

## Provenance

This crate was extracted from the `ship_of_theseus_rs` repository (https://github.com/raulmc/ship-of-theseus), specifically from:

- `crates/ship-core/src/learning_trainer.rs` - The original `NeuromorphicTrainer`
- `crates/ship-core/src/neuromorphic_core.rs` - The `apply_stdp` function and SNN engine
- `crates/ship-core/src/ai/researcher.rs` - NeuromorphicSnapshot data structures

The extraction focused on the proven crypto training loops that ran in production for GPU mining optimization, removing all:
- FPGA-specific export logic
- UI and agent components
- Vendored candle dependencies
- Old mining binaries

## Usage

```rust
use spikenaut_trainer::{SpikenautTrainer, TrainingConfig};
use neuromod::SpikingNetwork;
use spikenaut_telemetry::TelemetrySnapshot;

// Create trainer with default config
let mut trainer = SpikenautTrainer::new(TrainingConfig::default());
let mut network = SpikingNetwork::new();

// Train with telemetry snapshots
let telemetry_data: Vec<TelemetrySnapshot> = /* ... */;
let summary = trainer.run_session(&mut network, &telemetry_data);

println!("Trained {} steps, avg reward: {:.4}", 
         summary.steps_processed, summary.avg_reward);
```

## Integration

This crate is designed to integrate seamlessly with the Spikenaut workspace:

```toml
[dependencies]
spikenaut-trainer = { path = "../spikenaut-trainer" }
```

Use with `spikenaut-hybrid` for SNN/LLM fusion training:

```rust
// In hybrid_telemetry example
let reward = trainer.train_step(model.snn_mut(), &snapshot);
```

## Architecture

The training loop follows this closed-loop pattern:

1. **Telemetry Ingestion** - Hardware metrics from `spikenaut-telemetry`
2. **Reward Computation** - Mining efficiency via `spikenaut-reward`
3. **Neuromodulation** - Map reward to dopamine/cortisol
4. **Spike Encoding** - Convert telemetry to spike trains
5. **Network Step** - SNN integration and firing
6. **STDP Update** - Reward-modulated weight changes

## License

GPL-3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
