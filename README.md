# Rationality Measurement and Theory for Reinforcement Learning Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains empirical verification of our rationality measures and theoretical analysis. More details are in the following paper:

Kejiang Qian, Amos Storkey, Fengxiang He. Rationality Measurement and Theory for Reinforcement Learning Agents. [arXiv](https://arxiv.org/pdf/2602.04737)

## Empirically Testable Hypotheses

Our theory leads to the following hypotheses.

- **H1: Benefits of regularisations**: layer normalisation (LN), $\ell_2$ regularisation (L2), and weight normalisation (WN), can penalise hypothesis complexity.

- **H2: Benefits of domain randomisation**: improves robustness of reinforcement learning algorithms against distribution shifts across environments.

- **H3: Deficits of environment shifts**: larger environment shifts lead to worse rationality. 

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
├── experiment_1/           # Rational risk gap experiments (state distribution induced by policy pi)
│   ├── exp1_*_reg.sh
│   ├── exp2_*_domain_rand.sh
│   ├── exp3_*_env_level.sh
│   └── exp4_*_reg_intensity.sh
│
├── experiment_2/           # Special case: state distribution induced by optimal policy pi^*
│   ├── exp1_*_reg.sh
│   ├── exp2_*_domain_rand.sh
│   ├── exp3_*_env_level.sh
│   └── exp4_*_reg_intensity.sh
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
All results are available at [Google Drive](https://drive.google.com/drive/folders/1gcsWy8hSoQdPl1BA4DvekTO_fitSxcp3?usp=sharing).

The reproduction scripts are organised into two groups corresponding to two definitions of the **expected rational risk gap**:

- **`experiment_1/`** Standard rational risk gap experiments. The expected rational risk uses the state distribution $D_h^{\pi,\dagger}$ induced by the **evaluated policy** $\hat{\pi}$ in deployment.
- **`experiment_2/`** Special case where the expected rational risk uses the state distribution $\mathcal{D}_h^{*,\dagger}$ induced by the **optimal policy** $\pi^*$ in deployment.

The choice is controlled by the `--expected_rational_gap` flag (`"evaluated policy"` or `"optimal policy"`).

### Exp1 – Regularisation

```bash
bash experiment_1/exp1_taxi_reg.sh        # D_h^{pi,\dagger}
bash experiment_1/exp1_cliff_reg.sh
bash experiment_2/exp1_taxi_reg.sh        # D_h^{*,\dagger}
bash experiment_2/exp1_cliff_reg.sh
```

### Exp2 – Domain Randomisation

```bash
bash experiment_1/exp2_taxi_domain_rand.sh
bash experiment_1/exp2_cliff_domain_rand.sh
bash experiment_2/exp2_taxi_domain_rand.sh
bash experiment_2/exp2_cliff_domain_rand.sh
```

### Exp3 – Environment Level

```bash
bash experiment_1/exp3_taxi_env_level.sh
bash experiment_1/exp3_cliff_env_level.sh
bash experiment_2/exp3_taxi_env_level.sh
bash experiment_2/exp3_cliff_env_level.sh
```

### Exp4 – Regularisation Intensity

```bash
bash experiment_1/exp4_taxi_reg_intensity.sh
bash experiment_1/exp4_cliff_reg_intensity.sh
bash experiment_2/exp4_taxi_reg_intensity.sh
bash experiment_2/exp4_cliff_reg_intensity.sh
```

Results will be saved to:

```
logs/{env}/{experiment}/
```

## Citation

If you use this code in your research, please cite:

```
@article{qian2025rationality,
  title={Rationality Measurement and Theory for Reinforcement Learning Agents},
  author={Qian, Kejiang and Storkey, Amos and He, Fengxiang},
  year={2025}
}

