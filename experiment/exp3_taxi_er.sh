#!/usr/bin/env bash
set -e

echo "Starting experiments..."

echo "Run experiments of reg baseline"
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0 --horizon 500 --seed 3 --experiment exp3_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0 --horizon 500 --seed 1 --experiment exp3_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0 --horizon 500 --seed 2 --experiment exp3_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0 --horizon 500 --seed 4 --experiment exp3_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0 --horizon 500 --seed 5 --experiment exp3_reg --num_episodes 2000 --env taxi --variable baseline

echo "Run experiments of env_randomization"
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0  --horizon 500 --mix_kernels 3 --mix_eps_low 0.05 --mix_eps_high 0.3 --env_randomization --seed 3 --experiment exp3_reg --num_episodes 2000 --env taxi --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0  --horizon 500 --mix_kernels 3 --mix_eps_low 0.05 --mix_eps_high 0.3 --env_randomization --seed 1 --experiment exp3_reg --num_episodes 2000 --env taxi --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0  --horizon 500 --mix_kernels 3 --mix_eps_low 0.05 --mix_eps_high 0.3 --env_randomization --seed 2 --experiment exp3_reg --num_episodes 2000 --env taxi --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0  --horizon 500 --mix_kernels 3 --mix_eps_low 0.05 --mix_eps_high 0.3 --env_randomization --seed 4 --experiment exp3_reg --num_episodes 2000 --env taxi --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --eps_infer 0 --alpha_d0 0  --horizon 500 --mix_kernels 3 --mix_eps_low 0.05 --mix_eps_high 0.3 --env_randomization --seed 5 --experiment exp3_reg --num_episodes 2000 --env taxi --variable envrnd_train_25

echo "All experiments finished."
