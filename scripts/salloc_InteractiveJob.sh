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
        # salloc --time=01:00:00 --cpus-per-task=8 --mem-per-cpu=8G --account=def-training-wa

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














############ JUPYTER
# [guest187@gra806 ~]$ module load python/3.9
# thon/bin/activate[guest187@gra806 ~]$ source /home/guest187/hackathon/bin/activate
# (hackathon) [guest187@gra806 ~]$ srun $VIRTUAL_ENV/bin/notebook.sh
# [W 05:11:00.979 NotebookApp] Error loading server extension jupyter_lsp
#     Traceback (most recent call last):
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/notebook/notebookapp.py", line 2050, in init_server_extensions
#         func(self)
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyter_lsp/serverextension.py", line 76, in load_jupyter_server_extension
#         nbapp.io_loop.call_later(0, initialize, nbapp, virtual_documents_uri)
#     AttributeError: 'NotebookApp' object has no attribute 'io_loop'
# [W 05:11:01.202 NotebookApp] Error loading server extension jupyterlab
#     Traceback (most recent call last):
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/notebook/notebookapp.py", line 2050, in init_server_extensions
#         func(self)
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab/serverextension.py", line 6, in load_jupyter_server_extension
#         from .labapp import LabApp
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab/labapp.py", line 15, in <module>
#         from jupyterlab_server import (
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab_server/__init__.py", line 4, in <module>
#         from .app import LabServerApp
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab_server/app.py", line 11, in <module>
#         from .handlers import LabConfig, add_handlers
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab_server/handlers.py", line 18, in <module>
#         from .settings_handler import SettingsHandler
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab_server/settings_handler.py", line 14, in <module>
#         from .settings_utils import SchemaHandler, get_settings, save_settings
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab_server/settings_utils.py", line 14, in <module>
#         from .translation_utils import DEFAULT_LOCALE, L10N_SCHEMA_NAME, is_valid_locale
#       File "/home/guest187/hackathon/lib/python3.9/site-packages/jupyterlab_server/translation_utils.py", line 17, in <module>
#         import entrypoints
#     ModuleNotFoundError: No module named 'entrypoints'
# [I 05:11:01.219 NotebookApp] Loading lmod extension
# [I 05:11:01.234 NotebookApp] Serving notebooks from local directory: /home/guest187
# [I 05:11:01.234 NotebookApp] Jupyter Notebook 6.5.4 is running at:
# [I 05:11:01.234 NotebookApp] http://gra806.graham.sharcnet:8888/?token=443be9b532d582606e66bb03749bb4812051bd7e4ed2a55c
# [I 05:11:01.234 NotebookApp]  or http://127.0.0.1:8888/?token=443be9b532d582606e66bb03749bb4812051bd7e4ed2a55c
# [I 05:11:01.235 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
# [C 05:11:01.242 NotebookApp] 
    
#     To access the notebook, open this file in a browser:
#         file:///home/guest187/.local/share/jupyter/runtime/nbserver-9163-open.html
#     Or copy and paste one of these URLs:
#         http://gra806.graham.sharcnet:8888/?token=443be9b532d582606e66bb03749bb4812051bd7e4ed2a55c
#      or http://127.0.0.1:8888/?token=443be9b532d582606e66bb03749bb4812051bd7e4ed2a55c


##### DATA PREP #######
# python $git/data_prep_OptiNet.py --preproc_set training --task data_prep --data $data_dir --data_grp ATr --gpus 0
# nvidia-smi

##### TRAINING nnUNET #######
# srun --ntasks-per-node=2 python3 /home/guest187/BrainHackProject/nnUNet/main.py --data $data_dirTr --results $results_dirTr --ckpt_store_dir $ckpt_store --brats --depth 6 --filters 64 96 128 192 256 384 512 --scheduler --learning_rate 0.0005 --epochs 100 --fold 1 --amp --gpus 2 --task 11 --save_ckpt --nfolds 10 --exec_mode train

python -m ipykernel install --user --name PythonKern --display-name "Python 3.9 Kernel"