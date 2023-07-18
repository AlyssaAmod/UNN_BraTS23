#!/bin/bash

########## USE THE FOLLOWING COMMANDS TO CREATE AN INTERACTIVE SESSSION TO TEST CODE #########
# 
###### salloc --account=def-training-wa --time=hh:mm:ss --cpus-per-task=[3-22] --mem-per-cpu=[8-12G] --gres=gpu:t4:[1-3] ###########
######          other options include: --nodes[1-3], --gpus-per-task=t4:[1-3] --ntasks=[2?]                              ###########
###################################################################################################
###### EXAMPLE USES:
        # salloc --time=3:0:0 --cpus-per-task=12 --mem-per-cpu=8G --ntasks=2 --account=def-training-wa --nodes 2
        # salloc --time=0:10:00 --gpus-per-node=t4:2 --cpus-per-task=8 --mem=64G --account=def-training-wa --nodes 2
        # salloc --time=01:00:00 --gpus-per-node=t4:2 --cpus-per-task=6 --mem-per-cpu=8G --account=def-training-wa
        # salloc --time=1:00:00 --gpus-per-node=v100:1 --cpus-per-task=3 --mem=64G --ntasks=3 --account=def-training-wa
        # salloc --time=01:00:00 --cpus-per-task=8 --mem=128G --account=def-training-wa

module load python/3.9
source /home/guest187/hackathon/bin/activate
data_dir=/scratch/guest187/Data/train_all
results_dir=/scratch/guest187/Data/train_all/results
chkpt_store=/scratch/guest187/Data/train_all/checkpoints
git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts

##### DATA PREP #######
# python $git/data_prep_OptiNet.py --preproc_set training --task data_prep --data $data_dir --data_grp ATr --gpus 0
# nvidia-smi

##### TRAINING nnUNET #######
# srun --ntasks-per-node=2 python3 /home/guest187/BrainHackProject/nnUNet/main.py --data $data_dirTr --results $results_dirTr --ckpt_store_dir $ckpt_store --brats --depth 6 --filters 64 96 128 192 256 384 512 --scheduler --learning_rate 0.0005 --epochs 100 --fold 1 --amp --gpus 2 --task 11 --save_ckpt --nfolds 10 --exec_mode train