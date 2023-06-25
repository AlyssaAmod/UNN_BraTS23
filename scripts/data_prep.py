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
import random
import nibabel as nib
import numpy as np
import torch

from data_class import MRIDataset
import utils
from utils import get_main_args
from utils import extract_imagedata
from data_transforms import transforms_preproc
from data_transforms import apply_transforms

def prepare_nifty(data_dir, modalities, args):
    """ 
    This is the main data prepartion function. 
    It extracts the the image data from each volume and then stacks all modalities into one file.
    It then applies standard image preprocessing such as one hot encoding, realignment to RAS+ Z normalisation
    data_loader and trainer will work with these files.
    Input:
        path to directory containing folders of subject IDs
        list of modalities
    Output:
        JSON files:]
            subj_info == subject IDs & dir paths
            image_info == shape & resolution data per subject per modality
            dataset == modality keys, segmentation keys, image-label pairs per subj
        NifTI files:
            subjIDxxx-stk.nii.gz == stacked nifti img data 
            subjIDxxx-lbl.nii.gz == seg mask img data

    """
    img_pth, seg_pth, subjIDls, subj_dirls = [], [], [], []

    img_modality = []
    img_shapes = {}

    for subj in sorted(os.listdir(data_dir)):
        # run through each subjectID folder
        subj_dir = os.path.join(data_dir, subj)
        subjID = str(subj_dir)
        subjIDls.append(subjID)
        subj_dirls.append(subj_dir)
        SSA = True if 'SSA' in subjID else False
        
        for file in os.listdir(subj_dir):
            # check folder contents
            if os.path.isfile(os.path.join(subj_dir, file)):
                # Save original segmentation mask (file path)
                if file.endswith("-seg.nii.gz"):
                    seg_pth.append(os.path.join(subj_dir, file))
                    #seg = load_nifty(data_dir, subjID, "seg")
                    seg = nib.load(os.path.join(data_dir, subjID,file))
                    seg_affine, seg_header = seg.affine, seg.header
                    seg = extract_imagedata(seg, "unit8")
                    #seg[vol == 4] = 3 --> not sure what this does yet
                    seg = nib.nifti1.Nifti1Image(seg, seg_affine, header=seg_header)
                    nib.save(seg, os.path.join(subj_dir, subjID + "-lbl.nii.gz"))
                elif [file.endswith(f"-{m}.nii.gz") for m in modalities]:
                    # Save original image (file path)
                    img_pth.append(os.path.join(subj_dir, file))
                    for idx, mname in enumerate(modalities):
                        # if os.path.join(subjID + f"-{mname}.nii.gz") in img_pth[idx]:
                        globals()[f'{mname}'] = nib.load(img_pth[idx])
                        img_modality.append(globals()[f'{mname}'])
                        img_shapes[f'{mname}']=img_modality[idx].shape
                    affine, header = img_modality[-1].affine, img_modality[-1].header
                    res = header.get_zooms()
                    imgs = np.stack([extract_imagedata(img_modality[m]) for m, mod in enumerate(modalities)], axis=-1)
                    imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
                    nib.save(imgs, os.path.join(subj_dir, subjID + "-stk.nii.gz"))
        
        # save a few bits of info into a json    
        print("Saving shape & resolution data per subject")
        img_info = {
            "img_shapes": img_shapes,
            "res": res,
            "img_modalitypth": img_pth
        }
        with open("image_info.json", "a") as file:
            json.dump(img_info, file)
        # print("Saving SubjIDs")



def file_prep(data_dir, modalities, dataMode, train):
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
    stk_path, lbls_path = os.path.join(data_dir, f"images_orig-{dataMode}"), os.path.join(data_dir, f"labels_orig-{dataMode}")
    call(f"mkdir {stk_path}", shell=True)
    if train:
        call(f"mkdir {lbls_path}", shell=True)
    
    dirs = glob(os.path.join(data_dir, "BraTS*"))
    for d in dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        images, labels = [], []
        for f in files:
            if any(ignor_str in f for ignor_str in modalities):
                continue
            if "-lbl" in f:
                labels.append(f)
                call(f"cp {f} {stk_path}", shell=True)
            else:
                images.append(f) 
                call(f"cp {f} {lbls_path}", shell=True)
    if train == "training":
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    
    # ********** NEED TO CHOOSE HOW WE WANT TO SAVE THE LABELS DICT *********
    #labels_dict = {"0": "background", "1": "NCR - necrotic tumor core", "2": "ED - peritumoral edematous/invaded tissue", "3": "ET - GD-enhancing tumor"}
    modality = {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}
    labels_dict = {"0": "background", "1": "NCR", "2": "ED", "3": "ET"}

    # **********These path pairs are not needed for data_loader or training--> this is for incase it is needed
    images, labels = glob(os.path.join(stk_path, "*")), glob(os.path.join(lbls_path, "*"))
    images = sorted([img.replace(data_dir + "/", "") for img in images])
    labels = sorted([lbl.replace(data_dir + "/", "") for lbl in labels])
    if train:
        key = "training"
        data_pairs_fold = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs_fold = [{"image": img} for img in images]

    # sAve some json files for dataloading
    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs}
    with open(os.path.join(data_dir, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

    datasetFold = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs
    }
    with open(os.path.join(data_dir, "datasetFold.json"), "w") as outfile:
        json.dump(datasetFold, outfile)


def preprocess_data(data_dir, img, mask, args):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
    '''
    # img is still a list of arrays of the 4 modalities from data files
    # mask is 3d array

    # return img as list of arrays, and mask as before
    outpath = os.path.join(data_dir, args.datasets + "_prepoc")
    call(f"mkdir {outpath}", shell=True)

    metadata_path = os.path.join(data_dir, "dataset.json")
    metadata = json.load(open(metadata_path), "r")
    pair = {metadata["image"], metadata["label"]}
    apply_trans, preproc_trans = transforms_preproc(pair)

    apply_transforms(pair, apply_trans)

    if args.exec_mode == "val":
        dataset_json = json.load(open(metadata_path, "r"))
        dataset_json["val"] = dataset_json["training"]
        with open(metadata_path, "w") as outfile:
            json.dump(dataset_json, outfile)

    return img, mask

def main():
    args = get_main_args()
    modalities = args.modal
    data_dir = args.data
    # origData = MRIDataset(args.data, args.task, modalities=modalities)
    # img_pth, seg_pth = origData.get_paths()
    # #subj_dir, subjID, SSA = origData.get_subj_info()

    
    
    print("Generating stacked nifti files.")
    startT = time.time()
    prepare_nifty(data_dir, modalities, args)
    print("Loaded all nifti files and saved image data \nSaving a copy to images and labels folders")
    train = True if args.preproc_set == 'training' else False
    file_prep(data_dir, modalities, args.data_grp, train)
    endT = time.time()
    print(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    # startT = time.time()
    # print("Beginning Preprocessing.")

    # run_parallel(preprocess_data(origData.data_dir, args))
    # end = time.time()
    # print(f"Data Processing complete. Total time taken: {(end - start):.2f}")

if __name__=='__main__':
    main()