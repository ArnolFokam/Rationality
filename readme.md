# Rationality Measurement and Theory for Reinforcement Learning Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements experiments for analysing the **rationality gap** of reinforcement learning agents under training–inference distribution shift.  
We focus on DQN-based learners in tabular-style environments (Taxi-v3 and CliffWalking) and study how regularisation and domain randomisation affect rational behaviour.

---

## Project Structure

```
Rationality/
├── src/
│   ├── env/                # Customised Taxi & CliffWalking environments
│   │   ├── taxi.py
│   │   └── cliffwalking.py
│   ├── model/              # DQN implementation
│   │   └── DQN.py
│   ├── utils/              # Logger & helper functions
│   ├── regularisers.py     # Regularisation modules
│   └── runners.py          # Training / evaluation pipeline
│
├── experiment/             # Reproduction scripts
│   ├── exp1_*_reg.sh
│   ├── exp2_*_domain_rand.sh
│   └── exp3_*_env_level.sh
│
└── train.py                # Main entry
```

---

## Installation

```bash
conda create -n rationality python=3.10
conda activate rationality

pip install torch gym numpy pandas matplotlib
```

---

## Quick Start

### Train DQN on Taxi

```bash
python train.py \
  --env taxi \
  --episodes 2000 \
  --regulariser ln
```

### Train on CliffWalking with domain randomisation

```bash
python train.py \
  --env cliffwalking \
  --eps_train 0.3
```

---

## Experiments Reproduction

### Exp1 – Regularisation

```bash
bash experiment/exp1_taxi_reg.sh
bash experiment/exp1_cliff_reg.sh
```

### Exp2 – Domain Randomisation

```bash
bash experiment/exp2_taxi_domain_rand.sh
bash experiment/exp2_cliff_domain_rand.sh
```

### Exp3 – Environment Level

```bash
bash experiment/exp3_taxi_env_level.sh
bash experiment/exp3_cliff_env_level.sh
```

Results will be saved to:

```
logs/{env}/{experiment}/
```

## Citation

If you use this code in your research, please cite:

```
@article{,
  title={Rationality Gap in Reinforcement Learning},
  author={...},
  year={2025}
}
```
