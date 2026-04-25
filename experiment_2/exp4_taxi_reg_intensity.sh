#!/usr/bin/env bash
set -e

echo "Starting experiments..."

echo "Run experiments of l2_coef=1e-7"
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-7 --seed 3 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-7
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-7 --seed 1 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-7
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-7 --seed 2 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-7
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-7 --seed 4 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-7
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-7 --seed 5 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-7

echo "Run experiments of l2_coef=1e-6"
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-6 --seed 3 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-6
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-6 --seed 1 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-6
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-6 --seed 2 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-6
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-6 --seed 4 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-6
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-6 --seed 5 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-6

echo "Run experiments of l2_coef=1e-5"
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-5 --seed 3 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-5
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-5 --seed 1 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-5
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-5 --seed 2 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-5
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-5 --seed 4 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-5
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-5 --seed 5 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-5

echo "Run experiments of l2_coef=1e-4"
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-4 --seed 3 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-4
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-4 --seed 1 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-4
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-4 --seed 2 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-4
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-4 --seed 4 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-4
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-4 --seed 5 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-4

echo "Run experiments of l2_coef=1e-3"
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-3 --seed 3 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-3
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-3 --seed 1 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-3
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-3 --seed 2 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-3
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-3 --seed 4 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-3
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-3 --seed 5 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-3

echo "Run experiments of l2_coef=1e-1"
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-1 --seed 3 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-1
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-1 --seed 1 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-1
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-1 --seed 2 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-1
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-1 --seed 4 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-1
python ../train.py --algo dqn --eps_train 0.1  --l2_coef 1e-1 --seed 5 --experiment exp_reg_intensity --num_episodes 2500 --env taxi --variable l2_1e-1


echo "All experiments finished."
