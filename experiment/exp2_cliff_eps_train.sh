set -e

echo "Starting experiments..."

echo "Run experiments of eps_train 0.1"
python ../train.py --eps_train 0.10 --eps_infer 0 --alpha_d0 0.0 --seed 1 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_01
python ../train.py --eps_train 0.10 --eps_infer 0 --alpha_d0 0.0 --seed 2 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_01
python ../train.py --eps_train 0.10 --eps_infer 0 --alpha_d0 0.0 --seed 3 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_01
python ../train.py --eps_train 0.10 --eps_infer 0 --alpha_d0 0.0 --seed 4 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_01
python ../train.py --eps_train 0.10 --eps_infer 0 --alpha_d0 0.0 --seed 5 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_01

echo "Run experiments of eps_train 0.3"
python ../train.py --eps_train 0.3 --eps_infer 0 --alpha_d0 0.0 --seed 1 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_03
python ../train.py --eps_train 0.3 --eps_infer 0 --alpha_d0 0.0 --seed 2 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_03
python ../train.py --eps_train 0.3 --eps_infer 0 --alpha_d0 0.0 --seed 3 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_03
python ../train.py --eps_train 0.3 --eps_infer 0 --alpha_d0 0.0 --seed 4 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_03
python ../train.py --eps_train 0.3 --eps_infer 0 --alpha_d0 0.0 --seed 5 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_03

echo "Run experiments of eps_train 0.5"
python ../train.py --eps_train 0.5  --eps_infer 0 --alpha_d0 0.0 --seed 1 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_05
python ../train.py --eps_train 0.5  --eps_infer 0 --alpha_d0 0.0 --seed 2 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_05
python ../train.py --eps_train 0.5  --eps_infer 0 --alpha_d0 0.0 --seed 3 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_05
python ../train.py --eps_train 0.5  --eps_infer 0 --alpha_d0 0.0 --seed 4 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_05
python ../train.py --eps_train 0.5  --eps_infer 0 --alpha_d0 0.0 --seed 5 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_05

echo "Run experiments of eps_train 0.7"
python ../train.py --eps_train 0.7 --eps_infer 0 --alpha_d0 0.0 --seed 1 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_07
python ../train.py --eps_train 0.7 --eps_infer 0 --alpha_d0 0.0 --seed 2 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_07
python ../train.py --eps_train 0.7 --eps_infer 0 --alpha_d0 0.0 --seed 3 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_07
python ../train.py --eps_train 0.7 --eps_infer 0 --alpha_d0 0.0 --seed 4 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_07
python ../train.py --eps_train 0.7 --eps_infer 0 --alpha_d0 0.0 --seed 5 --experiment exp2_eps_train --num_episodes 400 --env cliffwalking --variable eps_train_07

echo "All experiments finished."
