set -e

echo "Starting experiments..."

echo "Run experiments of alpha_d0 0"
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.0 --seed 1 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.0 --seed 2 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.0 --seed 3 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.0 --seed 4 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.0 --seed 5 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking

echo "Run experiments of alpha_d0 0.1"
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.10 --seed 1 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_01
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.10 --seed 2 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_01
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.10 --seed 3 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_01
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.10 --seed 4 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_01
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.10 --seed 5 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_01

echo "Run experiments of alpha_d0 0.3"
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.3 --seed 1 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_03
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.3 --seed 2 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_03
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.3 --seed 3 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_03
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.3 --seed 4 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_03
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.3 --seed 5 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_03

echo "Run experiments of alpha_d0 0.5"
python ../train.py --eps_train 0  --eps_infer 0 --alpha_d0 0.5 --seed 1 --horizon 80  --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_05
python ../train.py --eps_train 0  --eps_infer 0 --alpha_d0 0.5 --seed 2 --horizon 80  --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_05
python ../train.py --eps_train 0  --eps_infer 0 --alpha_d0 0.5 --seed 3 --horizon 80  --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_05
python ../train.py --eps_train 0  --eps_infer 0 --alpha_d0 0.5 --seed 4 --horizon 80  --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_05
python ../train.py --eps_train 0  --eps_infer 0 --alpha_d0 0.5 --seed 5 --horizon 80  --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_05

echo "Run experiments of alpha_d0 0.7"
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.7 --seed 1 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_07
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.7 --seed 2 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_07
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.7 --seed 3 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_07
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.7 --seed 4 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_07
python ../train.py --eps_train 0 --eps_infer 0 --alpha_d0 0.7 --seed 5 --horizon 80 --experiment exp1_alpha_d0 --num_episodes 1000 --env cliffwalking --variable alpha_d0_07

echo "All experiments finished."

