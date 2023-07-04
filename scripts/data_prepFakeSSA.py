""" This script is to prepare the provided data set for pre-processing and then run all pre-processing.
It will take as input a directory path to the training data and apply the following:
    1. Read in the dataset folder structure
    2. Store variables
        subjID = subject IDs
        img_dir = path to each imaging modality
        lbl dir = path to corresponding segmentation mask
    3. Load nifty file for each modality
    4. Extract voxel intensity values, header information and affine matrix
        Stack voxel data from each modality into 1 tensor
        cropping out background
        Add an extra channel for one hot encoding??
    5. Save the following files:
        images/subjIDxxx-stk.nii.gz = stacked modalities output into a nifti file in an images folder
        labels/subjIDxxx-lbl.nii.gz = segmentation mask
    6. Create and save json file that contains a dictionary of dictionaries and lists:
        A dictionary of dummy coding for seg labels as provided by BraTS
            "labels" : {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
        A dictionary of dummy coding for each modality
            "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
        A dictionary of dictionaries containing the image-label path pairs
            "training": [{"image": "images/subjIDxxx.nii.gz", "label": "labels/subjIDxxx_seg.nii.gz"}

Add noise defs for fake SSA data in an if 
"""

## Import key libraries
import os
from glob import glob
import json
import time
from subprocess import call
import logging
from joblib import Parallel, delayed

import nibabel as nib
import numpy as np
import torch

import torchio as tio

import utils.utils
from utils.utils import get_main_args
from data_transforms import transforms_preproc
from data_transforms import define_transforms

def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

def preprocess_data(transList):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
        # img is still a list of arrays of the 4 modalities from data files
    mask is 3d array

    return img as list of arrays, and mask as before
    '''
    args = get_main_args()
    data_dir = args.data
   
    outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
    call(f"mkdir -p {outpath}", shell=True)

    # Define the list of helper functions for the transformation pipeline
    transform_pipeline = transforms_preproc(args.target_shape)[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = glob(os.path.join(data_dir, "BraTS*"))
    for d in dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        for f in files:
            if "-stk.nii.gz" not in f and "-lbl.nii.gz" not in f:
                continue
            elif "-stk.nii.gz" in f:
                proc_img = nib.load(f)
                proc_img = get_data(proc_img)
                proc_img_t = (torch.from_numpy(proc_img)).to(device)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_img_t = trans(proc_img_t)
                np.save(os.path.join(os.path.dirname(f), str(d) + "-stk_FSSA.npy"), proc_img_t)
            elif "-lbl.nii.gz" in f:
                proc_lbl = nib.load(f)
                proc_lbl = get_data(proc_lbl)
                proc_lbl_t = (torch.from_numpy(proc_lbl)).to(device)
                proc_lbl_t = torch.unsqueeze(proc_lbl_t, axis=0)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_lbl_t = trans(proc_lbl_t)
                np.save(os.path.join(os.path.dirname(f), str(d) + "-lbl_FSSA.npy"), proc_img_t)

def main():
    logging.basicConfig(filename='04-07_data_prep_fakeSSA.log', filemode='w', level=logging.DEBUG)


    logging.info("Beginning Preprocessing.")
    startT2 = time.time()

    data_transforms = define_transforms()
    transL=data_transforms['fakeSSA']
    run_parallel(preprocess_data, transL)
    
    end2= time.time()
    logging.info(f"Data Processing complete. Total time taken: {(end2 - startT2):.2f}")
    
if __name__=='__main__':
    main()