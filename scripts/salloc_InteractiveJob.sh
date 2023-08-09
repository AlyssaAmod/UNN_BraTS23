#!/bin/bash

########## USE THE FOLLOWING COMMANDS TO CREATE AN INTERACTIVE SESSSION TO TEST CODE ######### 
###### salloc --account=def-training-wa --time=hh:mm:ss --cpus-per-task=[3-22] --mem-per-cpu=[8-12G] --gres=gpu:t4:[1-3] ###########
######          other options include: --nodes[1-3], --gpus-per-task=t4:[1-3] --ntasks=[2?]                              ###########
###################################################################################################
###### EXAMPLE USES:
        # salloc --time=3:0:0 --cpus-per-task=12 --mem-per-cpu=8G --ntasks=2 --account=def-training-wa --nodes 2
        # salloc --time=0:10:00 --gpus-per-node=t4:2 --cpus-per-task=8 --mem=64G --account=def-training-wa --nodes 2
        # salloc --time=01:00:00 --gpus-per-node=t4:2 --cpus-per-task=6 --mem-per-cpu=8G --account=def-training-wa
        # salloc --time=1:00:00 --gpus-per-node=v100:1 --cpus-per-task=3 --mem=64G --ntasks=3 --account=def-training-wa
        # salloc --time=03:00:00 --cpus-per-task=12 --mem-per-cpu=8G --account=def-training-wa --gpus-per-node=t4:1

###############################################################
################### DIRECTORY PATHS ###########################
# data_dir=/scratch/guest187/Data/train_all
# results_dir=/scratch/guest187/Data/train_all/results
# chkpt_store=/scratch/guest187/Data/train_all/checkpoints
# git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts

# FOR MONAI trainer testing use
salloc --time=02:00:00 --gpus-per-node=t4:1 --cpus-per-task=6 --mem-per-cpu=8G --account=def-training-wa

module load python/3.9
source /home/guest187/hackathon/bin/activate

data_dir=/scratch/guest187/Data/train_all/train_data
results_dir=/scratch/guest187/Data/train_all/results/test_run
git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts


python $git/monai_trainer.py --seed 42 --data $data_dir --results $results_dir --epochs 2 --gpus 1 --run_name "tester" --data_used "SSA" --criterion "dice" --batch_size=2

#------------------ USING JUPYTER ----------------
salloc --time=03:00:00 --cpus-per-task=12 --mem-per-cpu=8G --account=def-training-wa --gpus-per-node=t4:1

salloc --time=03:00:00 --cpus-per-task=6 --mem-per-cpu=8G --account=def-training-wa --gpus-per-node=v100:1 --constraint=cascade,v100
module load python/3.9
source /home/guest187/hackathon/bin/activate

# I suggest using jupyter lab
srun $VIRTUAL_ENV/bin/jupyterlab.sh

## but you can also:
srun $VIRTUAL_ENV/bin/notebook.sh

## the following is the same regardless:
# once the above runs, it will give you a bunch of urls
# open local terminal and follow instructions from https://docs.alliancecan.ca/wiki/Advanced_Jupyter_configuration

# e.g. http://gra1156.graham.sharcnet:8888/lab?token=ef6a0f72cabe151aa7dbc808f0b52a2e13e7b35ca7c36a95
# In your local terminal type:
        # For mac/linux: sshuttle --dns -Nr <username>@<cluster>.computecanada.ca
        sshuttle --dns -Nr guest187@graham.computecanada.ca
                ## paste url that looks like this in your browser:
                http://gra1154.graham.sharcnet:8888/lab?token=e717e3ccab3c0664a46be3bd29fdfb047e9a6e9417bfac96

        # For windows  ssh -L 8888:<hostname:port> <username>@<cluster>.computecanada.ca
        ssh -L 8888:gra1162.graham.sharcnet:8888 guest187@graham.computecanada.ca
                # on chrome/firefox type: http://localhost:8888/?token=<token>
                http://localhost:8888/?token=b60f351f238d9abd066e5877b7fdb84096a45a50f22b69ea

                # OR this one which works better with jupyter lab
                http://127.0.0.1:8888/lab?token=1b3f59a629180f81f366eb98d0f0c3659f12a9b02d4b2b1

http://127.0.0.1:8888/lab?token=e717e3ccab3c0664a46be3bd29fdfb047e9a6e9417bfac96





