
# salloc --time=3:0:0 --cpus-per-task=12 --mem-per-cpu=8G --ntasks=2 --account=def-training-wa --nodes 2

module load python/3.9
source /home/guest187/hackathon/bin/activate
data_dir=/scratch/guest187/Data/train_all
results_dir=/scratch/guest187/Data/train_all/results
chkpt_store=/scratch/guest187/Data/train_all/checkpoints
git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts

##### DATA PREP #######
python $git/data_prep_OptiNet.py --preproc_set training --task data_prep --data $data_dir --data_grp ATr --gpus 0
# nvidia-smi

# TRAINING

# salloc --time=0:10:00 --gpus-per-node=t4:2 --cpus-per-task=8 --mem=64G --account=def-training-wa --nodes 2

# module load python/3.9
# source /home/guest187/hackathon/bin/activate
# data_dirTr=/scratch/guest187/Data/train_all_nnUNET/results/11_3d
# data_dirV=/scratch/guest187/Data/val_gli_nnUNET
# results_dirTr=/scratch/guest187/Data/train_all_nnUNET/results
# ckpt_store=/scratch/guest187/Data/train_all_nnUNET/results/checkpoints

# srun --ntasks-per-node=2 python3 /home/guest187/BrainHackProject/nnUNet/main.py --data $data_dirTr --results $results_dirTr --ckpt_store_dir $ckpt_store --brats --depth 6 --filters 64 96 128 192 256 384 512 --scheduler --learning_rate 0.0005 --epochs 100 --fold 1 --amp --gpus 2 --task 11 --save_ckpt --nfolds 10 --exec_mode train