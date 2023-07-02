
salloc --time=2:0:0 --gpus-per-node=t4:2 --cpus-per-task=16 --mem=16G --ntasks=2 --account=def-training-wa --nodes=2

module load python/3.9
source /home/guest187/hackathon/bin/activate
data_dir=/scratch/guest187/BraTS_Africa_data/ALL_TrainingData
results_dir=/scratch/guest187/BraTS_Africa_data/ALL_TrainingData/results
chkpt_store=/scratch/guest187/BraTS_Africa_data/ALL_TrainingData/checkpoints
# chkpt=/scratch/guest187/Hackathon/BraTS_2021_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/results_smlMdl/checkpoints/epoch=95-dice=89.61.ckpt
git=/home/guest187/AlyGit/UNN_BraTS23/scripts

nvidia-smi

# pip install --upgrade torchio
python3 $git/data_prep.py --task data_prep --data $data_dir --data_grp ATr --gpus 2
    # Run 27-06:
    #torchio not part of pre-installed hackathon package
    #albumentations not part of pre-installed hackathon package
    #several issues running through bash


# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120
# pip install -r /home/guest187/nnUNet/requirements.txt 
# python -c "import nvidia.dali"