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
from joblib import Parallel, delayed
import random
from utils import get_main_args
from utils import extract_imagedata

import nibabel as nib
import numpy as np
import torch
import torchio as tio

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
    imgs = np.stack([extract_imagedata(img_modality[m]) for m in modalities], axis=-1)
    imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
    nib.save(imgs, os.path.join(subj_dir + "-stk.nii.gz"))

    if os.path.exists(os.path.join(subj_dir, subjID + "-seg.nii.gz")):
            #seg = load_nifty(data_dir, subjID, "seg")
            seg = nib.load(os.path.join(subj_dir, subjID + "-seg.nii.gz"))
            seg_affine, seg_header = seg.affine, seg.header
            seg = extract_imagedata(seg, "unit8")
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
            if "-lbl" in f:
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
    with open(os.path.join(data_dir, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

    # **********These path pairs are not needed for data_loader or training--> this is for incase it is needed
    images, labels = glob(os.path.join(img_path, "*")), glob(os.path.join(lbl_path, "*"))
    images = sorted([img.replace(data_dir + "/", "") for img in images])
    labels = sorted([lbl.replace(data_dir + "/", "") for lbl in labels])
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
    with open(os.path.join(data_dir, "datasetFold.json"), "w") as outfile:
        json.dump(datasetFold, outfile)

def transforms_preproc(target_shape, pair):
    to_ras = tio.ToCanonical() # reorient to RAS+
    resample_t1space = tio.Resample(pair["image"], image_interpolation='nearest'), # target output space (ie. match T2w to the T1w space) 
    crop_pad = tio.CropOrPad(target_shape)
    # normalise = tio.ZNormalization()
    ohe = tio.OneHot(num_classes=4)
    mask = tio.Mask(masking_method=tio.LabelMap(pair["label"]))
    normalise_foreground = tio.ZNormalization(masking_method=lambda x: x > x.float().mean()) # threshold values above mean only, for binary mask
    preproc_trans = [to_ras, resample_t1space, ohe, mask, normalise_foreground]
    return preproc_trans

def data_preproc(args, target_spacing=None):
   
    print("Generating stacked nifti files.")
    start = time.time()
    run_parallel(prepare_nifty(args.data, args.modal))
    print("Loaded all nifti files and saved image data \nSaving a copy to images and labels folders")
    file_prep(args.data, task, args.modal, args.exec_mode)
    print(f"Image - label pairs created. Total time taken: {(end - start):.2f}")

    print("Beginning Preprocessing.")
    metadata_path = os.path.join(args.data, "dataset.json")
    metadata = json.load(open(metadata_path, "dataset.json"), "r"))
    pair = {metadata["image"], metadata["label"]}
    preproc_trans = transforms_preproc(args.target_shape, pair)
    apply_trans = tio.Compose(preproc_trans)

    if not args.exec_mode == "training":
        results = os.path.join(args.results, args.exec_mode, args.datasets + "_prepoc")
    else:
        results = os.path.join(args.results, args.datasets + "_prepoc")

    if args.exec_mode == "val":
        dataset_json = json.load(open(metadata_path, "r"))
        dataset_json["val"] = dataset_json["training"]
        with open(metadata_path, "w") as outfile:
            json.dump(dataset_json, outfile)

def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

def main():
    args = get_main_args()
    print(f"Preparing BraTS23 dataset from: {args.data}")
    start = time.time()
    data_preproc(args)
    print(f"Data Processing complete. Total time taken: {(end - start):.2f}")

if __name__=='__main__':
    main()