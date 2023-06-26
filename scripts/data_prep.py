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
from pathlib import Path
import tio

from data_class import MRIDataset
import utils
from utils import get_main_args
from utils import extract_imagedata
from data_transforms import transforms_preproc
from data_transforms import apply_transforms

def prepare_nifty(dataset):
    """ 
    This is the main data prepartion function. 
    It extracts the the image data from each volume and then stacks all modalities into one file.
    It then applies standard image preprocessing such as one hot encoding, realignment to RAS+ Z normalisation
    data_loader and trainer will work with these files.
    Input:
        dataset class
        args
        # OLD: 
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
    data_dir = dataset.data_dir
    subj_dirs = dataset.subj_dirs
    img_pth, seg_pth = dataset.get_paths()

    subjIDs = dataset.subjIDls

    modalities = dataset.modalities

    for subj in subj_dirs:
        pth = os.path.join(data_dir, subj)
        for root, dirs, files in os.walk(pth):
            for fileName in files:
                print("Entering for loop 3: \nSubj ", subj, "\nFile name ", fileName)
                if not fileName.endswith(".nii.gz"):
                    continue
                if fileName.endswith("-seg.nii.gz"):
                    seg = nib.load(os.path.join(root,fileName))
                    seg_affine, seg_header = seg.affine, seg.header
                    seg = extract_imagedata(seg, "unit8")
                    #seg[vol == 4] = 3 --> not sure what this does yet
                    seg = nib.nifti1.Nifti1Image(seg, seg_affine, header=seg_header)
                    nib.save(seg, os.path.join(root, subj + "-lbl.nii.gz"))
                else:
                    img_modality = []
                    img_shapes = {}
                    for m, mname in enumerate(modalities):
                        print("Entering for loop 4: ", m, mname)
                        globals()[f'{mname}'] = nib.load(os.path.join(root,fileName))
                        img_modality.append(globals()[f'{mname}'])
                        img_shapes[f'{mname}']=img_modality[m].shape
                    print(img_modality)
                    affine, header = img_modality[-1].affine, img_modality[-1].header
                    res = header.get_zooms()
                    imgs = np.stack([extract_imagedata(img_modality[m]) for m in range(len(img_modality))], axis=-1)
                    imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
                    nib.save(imgs, os.path.join(root, subj + "-stk.nii.gz"))
                             
    # save a few bits of info into a json 
                
                # img_info = {
                #     "img_shapes": img_shapes,
                #     "res": res,
                #     "img_modalitypth": img_pth}
                # print("Saving shape & resolution data per subject")
                # with open("image_info.json", "a") as file:
                #     json.dump(img_info, file)
               
    subj_info = {
        "subjIDs" : subjIDs
    }
    print("Saving SubjIDs")
    with open(os.path.join(data_dir+"subj_info.json"), "a") as file:
        json.dump(subj_info,file)


def file_prep(dataset, dataMode, train):
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

    data_dir = dataset.data_dir
    modalities = dataset.modalities
    stk_path, lbls_path = os.path.join(data_dir, f"images_orig-{dataMode}"), os.path.join(data_dir, f"labels_orig-{dataMode}")
    call(f"mkdir {stk_path}", shell=True)
    if train:
        call(f"mkdir {lbls_path}", shell=True)
    
    dirs = glob(os.path.join(data_dir, "BraTS*"))
    for d in dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        images, labels = [], []
        for f in files:
            if any(m in f for m in modalities) or '-seg' in f:
                continue
            if "-lbl" in f:
                labels.append(f)
                call(f"copy {f} {lbls_path}", shell=True)
            else:
                images.append(f) 
                call(f"copy {f} {stk_path}", shell=True)
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
        key: data_pairs_fold}
    with open(os.path.join(data_dir, "datasetFold.json"), "w") as outfile:
        json.dump(datasetFold, outfile)


def preprocess_data(dataset, args):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
    '''
    # img is still a list of arrays of the 4 modalities from data files
    # mask is 3d array

    # return img as list of arrays, and mask as before
    
    data_dir = dataset.data_dir
    subj_dirs = dataset.subj_dirs
    img_pth, seg_pth = dataset.get_paths()

    modalities = dataset.modalities

    outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
    call(f"mkdir {outpath}", shell=True)
    img = []
    mask = []

    to_ras = tio.ToCanonical() # reorient to RAS+
    resample_t1space = tio.Resample(img[0], image_interpolation='nearest') # target output space (ie. match T2w to the T1w space) 
    if args.target_shape != None:
        crop_pad = tio.CropOrPad(args.target_shape)
    one_hot_enc = tio.OneHot(num_classes=4)
    normalise_foreground = tio.ZNormalization(masking_method=lambda x: x > x.float().mean()) # threshold values above mean only, for binary mask
    masked = tio.Mask(masking_method=tio.LabelMap(mask))
    normalise = tio.ZNormalization()
        
    apply_trans = {
        'checkRAS' : to_ras,
        'resampleTot1' : resample_t1space,
        'oheZN' : tio.Compose([crop_pad, one_hot_enc, normalise_foreground]),
        'brainmask' : tio.Compose([crop_pad, masked, normalise])
        }
    
    imgs = []
    masks = []
    for subj in subj_dirs:
        pth = os.path.join(data_dir, subj)
        for root, dirs, files in os.walk(pth):
            for fileName in files:
                if not fileName.endswith(".nii.gz"):
                    continue
                elif fileName.endswith("-lbl.nii.gz"):
                    mask = nib.load(os.path.join(root,fileName))
                    mask = extract_imagedata(mask)
                    mask = apply_transforms(mask, apply_trans['checkRAS', 'oheZN'])
                    mask= np.save(os.path.join(outpath, subj + "-stk.npy"), mask)
                    masks.append(masks)
                elif fileName.endswith("-stk.nii.gz"):
                    img = nib.load(os.path.join(root,fileName))
                    img = extract_imagedata(img)
        
                    img = apply_transforms(img, apply_trans['checkRAS', 'oheZN'])
                    img = np.save(os.path.join(outpath, subj + "-lbl.npy"), img)
                    imgs.append(img)
        


    return imgs, masks

def main():
    args = get_main_args()
    modalities = args.modal
    data_dir = args.data
    task = args.task
    origData = MRIDataset(data_dir, task, modalities=modalities)
      
    # print("Generating stacked nifti files.")
    # startT = time.time()
    # prepare_nifty(origData)
    # print("Loaded all nifti files and saved image data \nSaving a copy to images and labels folders")
    # train = True if args.prepoc_set == 'training' else False
    # file_prep(origData, args.data_grp, train)
    # endT = time.time()
    # print(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    startT = time.time()
    print("Beginning Preprocessing.")
    preprocess_data(origData, args)
    # utils.run_parallel(preprocess_data(origData.data_dir, args))
    end = time.time()
    print(f"Data Processing complete. Total time taken: {(end - start):.2f}")

if __name__=='__main__':
    main()