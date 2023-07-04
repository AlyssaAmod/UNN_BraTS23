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




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)
    
# def run_parallel(func, *args):
#     return Parallel(n_jobs=-1)(delayed(func)(*arg) for arg in zip(*args))

def load_nifty(directory, example_id, suffix):
    return nib.load(os.path.join(directory, example_id + "-" + suffix + ".nii.gz"))


def load_channels(d, example_id):
    return [load_nifty(d, example_id, suffix) for suffix in ["t2f", "t1n", "t1c", "t2w"]]


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)


def prepare_nifty(d):
    example_id = d.split("/")[-1]
    flair, t1, t1ce, t2 = load_channels(d, example_id)
    affine, header = flair.affine, flair.header
    vol = np.stack([get_data(flair), get_data(t1), get_data(t1ce), get_data(t2)], axis=-1)
    vol = nib.nifti1.Nifti1Image(vol, affine, header=header)
    nib.save(vol, os.path.join(d, example_id + "-stk.nii.gz"))

    if os.path.exists(os.path.join(d, example_id + "-seg.nii.gz")):
        seg = load_nifty(d, example_id, "seg")
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        vol[vol == 4] = 3
        seg = nib.nifti1.Nifti1Image(vol, affine, header=header)
        nib.save(seg, os.path.join(d, example_id + "-lbl.nii.gz"))


def prepare_dirs(data, train):
    img_path, lbl_path = os.path.join(data, "images"), os.path.join(data, "labels")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(data, "BraTS*"))
    for d in dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        for f in files:
            if "t2f" in f or "t1n" in f or "t1c" in f or "t2w" in f:
                continue
            if "-seg" in f:
                call(f"cp {f} {lbl_path}", shell=True)
            else:
                call(f"cp {f} {img_path}", shell=True)


def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images", "*")), glob(os.path.join(data, "labels", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    modality = {"0": "t2f", "1": "t1n", "2": "t1c", "3": "t2w"}
    labels_dict = labels_dict = {"0": "background", "1": "NCR", "2": "ED", "3": "ET"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)


def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)


def prepare_dataset(data, train):
    print(f"Preparing BraTS21 dataset from: {data}")
    start = time.time()
    run_parallel(prepare_nifty, sorted(glob(os.path.join(data, "BraTS*"))))
    prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")


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
                np.save(os.path.join(os.path.dirname(f), str(d) + "-stk.npy"), proc_img_t)
            elif "-lbl.nii.gz" in f:
                proc_lbl = nib.load(f)
                proc_lbl = get_data(proc_lbl)
                proc_lbl_t = (torch.from_numpy(proc_lbl)).to(device)
                proc_lbl_t = torch.unsqueeze(proc_lbl_t, axis=0)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_lbl_t = trans(proc_lbl_t)
                np.save(os.path.join(os.path.dirname(f), str(d) + "-lbl.npy"), proc_img_t)

def main():
    logging.basicConfig(filename='04-07_data_prep_22h40.log', filemode='w', level=logging.DEBUG)
    args = get_main_args()
    utils.utils.set_cuda_devices(args)
      
    logging.info("Generating stacked nifti files.")
    startT = time.time()
    logging.info("Loaded all nifti files and saved image data")
    prepare_dataset(args.data, True)
    print("Finished!")
    endT = time.time()
    logging.info(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    logging.info("Beginning Preprocessing.")
    startT2 = time.time()
    # transL = ['checkRAS', 'CropOrPad', 'Znorm']
        # transform_pipeline = {
        # 'checkRAS' : to_ras,
        # 'CropOrPad' : crop_pad,
        # 'ohe' : one_hot_enc,
        # 'ZnormFore' : normalise_foreground,
        # 'MaskNorm' : masked,
        # 'Znorm': normalise
    # procArgs = (args, transL)
    data_transforms = define_transforms()
    transL=data_transforms['fakeSSA']
    run_parallel(preprocess_data, transL)
    
    end2= time.time()
    logging.info(f"Data Processing complete. Total time taken: {(end2 - startT2):.2f}")
    
if __name__=='__main__':
    main()