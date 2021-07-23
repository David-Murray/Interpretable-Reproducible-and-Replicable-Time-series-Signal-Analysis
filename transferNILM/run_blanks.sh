#!/bin/bash
START=0
END=549
STEP=1
echo "Countdown"
# echo $(source ~/anaconda3/bin/activate transfernilm)
# echo $(conda activate transfernilm)
 
for (( c=$START; c<=$END; c+=STEP ))
do
    echo $(/home/david/anaconda3/envs/transfernilm/bin/python seq2point_test.py --appliance_name 'washingmachine' --datadir './dataset_management/eco/' --trained_model_dir './trained_model' --save_results_dir './result' --transfer False --crop_dataset 1000000 --plot_results False --specific '_2' --blank $c --blanksize 50)
done
 
echo
echo "Boom!"