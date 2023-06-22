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
import random
import json
from glob import glob
import time
from subprocess import call
from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
def main():
    args = parser.parse_args()
    
    print(f"Preparing BraTS23 dataset from: {args.data}")
    start = time.time()
    # subj_dir = load_dir(args.data)
    run_parallel(prepare_nifty(args.data, args.modal))
    
    print("Loaded all nifti files and saved image data \nSaving a copy to images and labels folders")
    file_prep(args.data, args.modal, args.exec_mode)
    print("Image - label pairs created")
    end = time.time()
    print(f"Data ready for preprocessing. Total time taken: {(end - start):.2f}")
    
# Create args for pre-proc
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
#For file loading etc
parser.add_argument("--data", type=str, default="/data", help="Path to data directory")
parser.add_argument("--modal", tye=list, default=["t1c", "t1n", "t2f", "t2w"], help="List of modality abbreviations")
#For preprocessing stacked nifty files
parser.add_argument("--results", type=str, default="/data", help="Path for saving output directory")
parser.add_argument(
    "--exec_mode",
    type=str,
    default="training",
    choices=["training", "val", "test"],
    help="Mode for data preprocessing",
)
parser.add_argument("--ohe", action="store_true", help="Add one-hot-encoding for foreground voxels (voxels > 0)")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs for data preprocessing")

## ***** ADDED INTO ONE FUNCTION CALL --> CHECK WITH ALEX AND DELETE COMMENTED LINES *****
# # Read in the dataset folder structure
# def load_dir(directory):
#     data_dir = directory
#     subjID = sorted(os.listdir(data_dir))
#     print("You are working in :", data_dir, "Total subjects: ", len(subjID))
#     subj_dir = os.path.join(data_dir, subjID)
#     data = {
#         "subjID": subjID
#     }
#     with open("data_overview.json", "w") as file:
#         json.dump(data, file)
#     return subj_dir 

# def load_nifty(directory, subjID, m):
#     return nib.load(os.path.join(data_dir + subjID, subjID + f"-{m}.nii.gz"))

# Better to have fx for this than typing each time --> MAY WANT TO MOVE TO A UTILS.py
def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def prepare_nifty(directory, modalities):
    """ 
    This is the main data prepartion function. 
    It extracts the the image data from each volume and then stacks all modalities into one file.
    data_loader and trainer will work with these files.
    Input:
        path to directory containing folders of subject IDs
        list of modalities
    Output:
        subjIDxxx-stk.nii.gz == stacked nifti img data 
        subjIDxxx-lbl.nii.gz == seg mask img data
        data_overview.json == overview of subjIDs, 
                                shape and resolution of each modality per subj 
                                & path to original nii files of each modality
    """
# Create paths for each modality
    data_dir = directory
    subjID = sorted(os.listdir(data_dir))
    print("You are working in :", data_dir, "Total subjects: ", len(subjID))
    subj_dir = os.path.join(data_dir, subjID)
    img_modalitypth = [(os.path.join(subj_dir, subjID + f"-{m}.nii.gz")) for m in modalities]
        
# load modalities into a list, generate headers & affine
    img_modality = []
    img_shapes = {}
    for idx, mname in enumerate(modalities):
        globals()[f'{mname}'] = nib.load(img_modalitypth[idx])
        img_modality.append(globals()[f'{mname}'])
        img_shapes[f'{mname}']=img_modality[idx].shape
    affine, header = img_modality[-1].affine, img_modality[-1].header
    res = header.get_zooms()
    imgs = np.stack([get_data(img_modality[m]) for m in modalities], axis=-1)
    imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
    nib.save(imgs, os.path.join(subj_dir + "-stk.nii.gz"))

    if os.path.exists(os.path.join(subj_dir, subjID + "-seg.nii.gz")):
            #seg = load_nifty(data_dir, subjID, "seg")
            seg = nib.load(os.path.join(subj_dir, subjID + "-seg.nii.gz"))
            seg_affine, seg_header = seg.affine, seg.header
            seg = get_data(seg, "unit8")
            #seg[vol == 4] = 3 --> not sure what this does yet
            seg = nib.nifti1.Nifti1Image(seg, seg_affine, header=seg_header)
            nib.save(seg, os.path.join(subj_dir + "-lbl.nii.gz"))
    
# save a few bits of info into a json
        #print(f"Dimensions for modality {mname} is {img_modality[idx].shape}, with isotropic resolution of {hdrs[f'{mname}_img'].get_zooms()} ")
    print("Saving SubjIDs")
    subjDat = {
        "subjID": subjID
    }
    with open("data_overview.json", "w") as file:
        json.dump(subjDat, file)

    print("Saving shape & resolution data per subject")
    data = {
        "img_shapes": img_shapes,
        "res": res,
        "img_modalitypth": img_modalitypth
    }
    with open("data_overview.json", "a") as file:
        json.dump(data, file)


def file_prep(data_dir, modalities, train):
    """ 
    This an extra function to save a copy of the image data extracted from each volume.
    data_loader and trainer do not require these data as they are stored in the original subject folders as well

    Creates a json file with
        A dictionary of dummy coding for seg labels as provided by BraTS
            "labels" : {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
        A dictionary of dummy coding for each modality
            "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
        A dictionary of dictionaries containing the image-label path pairs
            "training": [{"image": "images/subjIDxxx.nii.gz", "label": "labels/subjIDxxx_seg.nii.gz"}
    """
    img_path, lbl_path = os.path.join(data_dir, "images"), os.path.join(data_dir, "labels")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(data_dir, "BraTS*"))
    for d in dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        images = []
        labels = []
        for f in files:
            if any(ignor_str in f for ignor_str in modalities):
                continue
            if "-seg" in f:
                labels.append(f)
                call(f"cp {f} {lbl_path}", shell=True)
            else:
                images.append(f) 
                call(f"cp {f} {img_path}", shell=True)
    
    modality = {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}
    # ********** NEED TO CHOOSE HOW WE WANT TO SAVE THE LABELS DICT *********
    #labels_dict = {"0": "background", "1": "NCR - necrotic tumor core", "2": "ED - peritumoral edematous/invaded tissue", "3": "ET - GD-enhancing tumor"}
    labels_dict = {"0": "background", "1": "NCR", "2": "ED", "3": "ET"}
    if train == "training":
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
            key = "test"
            data_pairs = [{"image": img} for img in images]
    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs
    }
    with open(os.path.join(args.data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

    # **********These path pairs are not needed for data_loader or training--> this is for incase it is needed
    images, labels = glob(os.path.join(img_path, "*")), glob(os.path.join(lbl_path, "*"))
    images = sorted([img.replace(args.data + "/", "") for img in images])
    labels = sorted([lbl.replace(args.data + "/", "") for lbl in labels])
    if train:
        key = "training"
        data_pairs_fold = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs_fold = [{"image": img} for img in images]

    datasetFold = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs
    }
    with open(os.path.join(args.data, "datasetFold.json"), "w") as outfile:
        json.dump(datasetFold, outfile)

def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)



if __name__=='__main__':
    main()