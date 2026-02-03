#!/usr/bin/env bash
set -e

echo "Starting experiments..."


echo "Run experiments of weightnorm"
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --weightnorm --seed 3 --experiment exp_reg --num_episodes 2000 --env taxi --variable wn_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --weightnorm --seed 1 --experiment exp_reg --num_episodes 2000 --env taxi --variable wn_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --weightnorm --seed 2 --experiment exp_reg --num_episodes 2000 --env taxi --variable wn_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --weightnorm --seed 4 --experiment exp_reg --num_episodes 2000 --env taxi --variable wn_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --weightnorm --seed 5 --experiment exp_reg --num_episodes 2000 --env taxi --variable wn_train

echo "Run experiments of reg baseline"
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 3 --experiment exp_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 1 --experiment exp_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 2 --experiment exp_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 4 --experiment exp_reg --num_episodes 2000 --env taxi --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 5 --experiment exp_reg --num_episodes 2000 --env taxi --variable baseline

echo "Run experiments of l2"
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --l2_coef 1e-5 --seed 3 --experiment exp_reg --num_episodes 2000 --env taxi --variable l2_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --l2_coef 1e-5 --seed 1 --experiment exp_reg --num_episodes 2000 --env taxi --variable l2_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --l2_coef 1e-5 --seed 2 --experiment exp_reg --num_episodes 2000 --env taxi --variable l2_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --l2_coef 1e-5 --seed 4 --experiment exp_reg --num_episodes 2000 --env taxi --variable l2_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --l2_coef 1e-5 --seed 5 --experiment exp_reg --num_episodes 2000 --env taxi --variable l2_train


echo "Run experiments of layernorm"
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --layernorm --seed 3 --experiment exp_reg --num_episodes 2000 --env taxi --variable ln_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --layernorm --seed 1 --experiment exp_reg --num_episodes 2000 --env taxi --variable ln_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --layernorm --seed 2 --experiment exp_reg --num_episodes 2000 --env taxi --variable ln_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --layernorm --seed 4 --experiment exp_reg --num_episodes 2000 --env taxi --variable ln_train
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --layernorm --seed 5 --experiment exp_reg --num_episodes 2000 --env taxi --variable ln_train

echo "All experiments finished."
