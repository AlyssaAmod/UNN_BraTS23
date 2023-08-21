from data_preparation import prepare_dataset
from main import main
from data_preprocessing.preprocessor import Preprocessor
from utils.utils import get_task_code

import os
import torch
import time

def run_inference(data_path, parameters, output_path, ckpt_path):

    # Preparing dataset
    prepare_dataset(data_path, False) # Testing
    print("Finished prepping all gli data!")

    parameters['ckpt_path'] = ckpt_path #?
    parameters['results'] = output_path #?

    # Preprocessing dataset
    start = time.time()
    Preprocessor(parameters).run()
    end = time.time()
    print(f"Pre-processing time: {(end - start):.2f}")

    # Making predictions and post-processing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main(parameters)