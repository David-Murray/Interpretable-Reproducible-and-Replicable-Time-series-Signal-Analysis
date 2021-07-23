#!/bin/bash
declare -a StringArray=("_1" "_2" "_3" "_4" "_5" "_rng_lock_1" "_rng_lock_6" "_rng_lock_3" "_rng_lock_4" "_rng_lock_5")
 
for val in ${StringArray[@]}; 
do
    echo $(/home/david/anaconda3/envs/transfernilm/bin/python seq2point_test.py --appliance_name 'washingmachine' --datadir './dataset_management/eco/' --trained_model_dir './trained_model' --save_results_dir './result' --transfer False --crop_dataset 6000000 --plot_results False --specific $val)
done
 
echo
echo "Boom!"