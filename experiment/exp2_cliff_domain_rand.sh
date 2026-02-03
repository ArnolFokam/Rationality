cliffwalking#!/usr/bin/env bash
set -e

echo "Starting experiments..."

echo "Run experiments of reg baseline"
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 3 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 1 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 2 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 4 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable baseline
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --seed 5 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable baseline

echo "Run experiments of env_randomization"
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --env_randomization --seed 3 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --env_randomization --seed 1 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --env_randomization --seed 2 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --env_randomization --seed 4 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable envrnd_train_25
python ../train.py --algo dqn --eps_train 0.25 --horizon 500 --env_randomization --seed 5 --experiment exp_domain_rand --num_episodes 4000 --env cliffwalking --variable envrnd_train_25
echo "All experiments finished."