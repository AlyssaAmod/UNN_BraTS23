
# salloc --time=2:0:0 --gpus-per-node=t4:2 --cpus-per-task=16 --mem=32G --ntasks=2 --account=def-training-wa --nodes 2

module load python/3.9
source /home/guest187/hackathon/bin/activate
data_dir=/scratch/guest187/Data/val_gli
results_dir=/scratch/guest187/Data/train_all/results
chkpt_store=/scratch/guest187/Data/train_all/checkpoints
git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts


# pip install --upgrade torchio
python3 $git/data_prep.py --data $data_dir --data_grp GV --preproc_set val --task data_prep --target_shape True --gpus 2
    # Run 27-06:
    #torchio not part of pre-installed hackathon package
    #albumentations not part of pre-installed hackathon package
    #several issues running through bash


# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120
# pip install -r /home/guest187/nnUNet/requirements.txt 
# python -c "import nvidia.dali"
nvidia-smi
