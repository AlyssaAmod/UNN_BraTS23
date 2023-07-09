
# salloc --time=3:0:0 --cpus-per-task=12 --mem-per-cpu=8G --ntasks=2 --account=def-training-wa --nodes 2

module load python/3.9
source /home/guest187/hackathon/bin/activate
data_dir=/scratch/guest187/Data/val_gli
results_dir=/scratch/guest187/Data/train_all/results
chkpt_store=/scratch/guest187/Data/train_all/checkpoints
git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts


# # pip install --upgrade torchio
# python3 $git/data_prep.py --data $data_dir --data_grp GV --preproc_set val --task data_prep --target_shape True --gpus 2
#     # Run 27-06:
#     #torchio not part of pre-installed hackathon package
#     #albumentations not part of pre-installed hackathon package
#     #several issues running through bash


module load python/3.9
source /home/guest187/hackathon/bin/activate
data_dir=/scratch/guest187/Data/train_all
results_dir=/scratch/guest187/Data/train_all/results
chkpt_store=/scratch/guest187/Data/train_all/checkpoints
git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts

##### DATA PREP #######
prep_scripts=/home/guest187/BraTS23_SSA/MainRun/prep_scripts
python $prep_scripts/data_prep.py --preproc_set training --task data_prep --data $data_dir --data_grp ATr --gpus 2
nvidia-smi