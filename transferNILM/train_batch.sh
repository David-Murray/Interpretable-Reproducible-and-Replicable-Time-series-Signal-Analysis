#!/bin/bash
declare -a StringArray=("_r1" "_r2" "_r3" "_r4" "_r5" "_rng_lock_r1" "_rng_lock_r2" "_rng_lock_r3" "_rng_lock_r4" "_rng_lock_r5")
 
for val in ${StringArray[@]}; 
do
    echo $(/home/david/anaconda3/envs/transfernilm/bin/python seq2point_test.py --appliance_name 'washingmachine' --datadir './dataset_management/eco/' --trained_model_dir './trained_model' --save_results_dir './result' --transfer False --crop_dataset 6000000 --plot_results False --specific $val)

    echo $(/home/david/anaconda3/envs/transfernilm/bin/python PYTHONHASHSEED=0 python seq2point_train.py --appliance_name 'washingmachine' --datadir './dataset_management/refit/' --save_dir './trained_model' --transfer_model False --crop_dataset 43470000 --n_epoch 50 --gpus 1
done
 
echo
echo "Boom!"