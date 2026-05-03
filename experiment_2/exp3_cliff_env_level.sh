#!/usr/bin/env bash
set -e

echo "Starting experiments..."

echo "Run experiments of eps_train 0.1"
python ../train.py --eps_train 0.10 --seed 1 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_01 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.10 --seed 2 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_01 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.10 --seed 3 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_01 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.10 --seed 4 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_01 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.10 --seed 5 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_01 --expected_rational_gap "optimal policy"

echo "Run experiments of eps_train 0.3"
python ../train.py --eps_train 0.3 --seed 1 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_03 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.3 --seed 2 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_03 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.3 --seed 3 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_03 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.3 --seed 4 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_03 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.3 --seed 5 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_03 --expected_rational_gap "optimal policy"

echo "Run experiments of eps_train 0.5"
python ../train.py --eps_train 0.5  --seed 1 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_05 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.5  --seed 2 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_05 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.5  --seed 3 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_05 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.5  --seed 4 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_05 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.5  --seed 5 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_05 --expected_rational_gap "optimal policy"

echo "Run experiments of eps_train 0.7"
python ../train.py --eps_train 0.7 --seed 1 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_07 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.7 --seed 2 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_07 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.7 --seed 3 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_07 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.7 --seed 4 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_07 --expected_rational_gap "optimal policy"
python ../train.py --eps_train 0.7 --seed 5 --experiment exp_environment_level --num_episodes 2500 --env cliffwalking --variable eps_train_07 --expected_rational_gap "optimal policy"

echo "All experiments finished."
