from data_preparation import prepare_dataset
from main import main
from data_preprocessing.preprocessor import Preprocessor
from utils.utils import get_task_code
from postprocessing import prepare_predictions, to_lbl

import os
import glob
import torch
import time

def run_inference(data_path, parameters):

    # Preparing dataset
    prepare_dataset(data_path, False) # Testing
    print("Finished prepping all gli data!")

    # Preprocessing dataset
    start = time.time()
    Preprocessor(parameters).run()
    end = time.time()
    print(f"Pre-processing time: {(end - start):.2f}")

    # Making predictions and post-processing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for ckpt in os.listdir(parameters['ckpts_path']):
        ckpt_path = os.path.join(parameters['ckpts_path'], ckpt)
        main(parameters, ckpt_path)

    # Post processing : Ensembling + To_label

    os.makedirs(os.path.join(parameters["results"], "predictions"))
    preds = sorted(glob(f"/results/predictions*"))
    examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
    print("Preparing final predictions")
    for e in examples:
        prepare_predictions(e)
    print("Finished!")