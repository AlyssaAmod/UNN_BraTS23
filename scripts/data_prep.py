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
import torchio as tio
import torchvision.transforms as transforms


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
    subj_dir_pths, subj_dirs =(dataset.get_subj_info())
    img_pth, seg_pth = dataset.get_paths()[0:2]

    modalities = dataset.modalities
    img_shapes = {}
    res = {}
    for subj in subj_dirs:
        img_modality = []
        for i in range(dataset.len(img_pth)):
            if subj not in img_pth[i]:
                break
            print("subjID = ",subj)
            for root, dirs, files in os.walk(img_pth[i]):
                print("Entering subject directory: ", os.path.basename(root), "\nFile name ", files[0])
                ext_dict_modal = {**{f"-{m}.nii.gz": img_modality for m in modalities}}
                for ext, list_to_append in ext_dict_modal.items():
                    if file.endswith(ext):
                        mod = ext.split("-")[-2]
                        globals()[f'{mod}'] = nib.load(img_pth[i])
                        #img_modality.append(globals()[f'{mname}'])
                        list_to_append.append(globals()[f'{mod}'])
                        print((globals()[f'{ext}']).shape)
                        img_shapes[f'{os.path.basename(root)}_{mod}']=list_to_append[-1].shape
        print(img_modality)
        affine, header = img_modality[-1].affine, img_modality[-1].header
        res[f'{os.path.basename(root)}_RES']=header.get_zooms()
        imgs = np.stack([extract_imagedata(img_modality[m]) for m in range(len(img_modality))], axis=-1)
        imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
        nib.save(imgs, os.path.join(root, subj + "-stk.nii.gz"))

        for i in range(dataset.len(seg_pth)):
            if subj not in seg_pth[i]:
                break
            for root, dirs, files in os.walk(seg_pth[i]):
                print("Entering subject directory: ", os.path.basename(root), "\nFile name ", files[0])
                if not files[0].endswith("-seg.nii.gz"):
                    break
                seg = nib.load(seg_pth[i])
                seg_affine, seg_header = seg.affine, seg.header
                seg = extract_imagedata(seg, "unit8")
                #seg[vol == 4] = 3 --> not sure what this does yet
                seg = nib.nifti1.Nifti1Image(seg, seg_affine, header=seg_header)
                print(seg.shape)
                nib.save(seg, os.path.join(root,  subj + "-lbl.nii.gz"))
                             
    # save a few bits of info into a json 
    img_info = {
        "img_shapes": img_shapes,
        "res": res,
        "img_modalitypth": img_pth}
    print("Saving shape & resolution data per subject")
    with open("image_info.json", "a") as file:
        json.dump(img_info, file)
               
    subj_info = {
        "subjIDs" : subj_dirs
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
    stk, lbl = dataset.get_paths()[2:2]
    subj_dirs = dataset.get_subj_info()[1]
    
    stk_path, lbls_path = os.path.join(data_dir, f"images_orig-{dataMode}"), os.path.join(data_dir, f"labels_orig-{dataMode}")
    call(f"mkdir {stk_path}", shell=True)
    if train:
        call(f"mkdir {lbls_path}", shell=True)
    
    imagesF, labelsF = [], []

    file_ext_dict2 = {
        "-lbl.nii.gz": labelsF,
        "-stk.nii.gz": imagesF}

    for subj in subj_dirs:
        # for i in range(dataset.len(stk)):
        #     for j in range(dataset.len(lbl)):
        #         if subj not in stk[i] or lbl[j]:
        #             break
        #         print("subjID = ",subj)
        #         for root, dirs, files in os.walk(stk[i]):
        #             for file in files:
        #                 if os.path.isfile(os.path.join(root, file)) :
        #                     for ext, list_to_append in file_ext_dict2.items():
        #                         if file.endswith(ext):
        #                         #print(file_pth)
        #                         list_to_append.append(os.path.join(root, file))
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if not f.endswith("-lbl.nii.gz") and not f.endswith("-stk.nii.gz"):
                    break
                if "-lbl" in f:
                    labelsF.append(f)
                    call(f"copy {f} {lbls_path}", shell=True)
                else:
                    imagesF.append(f) 
                    call(f"copy {f} {stk_path}", shell=True)
    if train == "training":
        key = "training"
        data_pairs = [{"image": imgF, "label": lblF} for (imgF, lblF) in zip(imagesF, labelsF)]
    else:
        key = "test"
        data_pairs = [{"image": imgF} for imgF in imagesF]

    modality = {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}
    labels_dict = {"0": "background", "1": "NCR", "2": "ED", "3": "ET"}

    # **********These path pairs are not needed for data_loader or training--> this is for incase it is needed
    images, labels = glob(os.path.join(stk_path, "*")), glob(os.path.join(lbls_path, "*"))
    images = sorted([img.replace(data_dir + "/", "") for img in images])
    labels = sorted([lbl.replace(data_dir + "/", "") for lbl in labels])
    if train == "training":
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


def preprocess_data(dataset, args, transList):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
    '''
    # img is still a list of arrays of the 4 modalities from data files
    # mask is 3d array

    # return img as list of arrays, and mask as before
    import itertools

    data_dir = dataset.data_dir
    
    subj_dirs = dataset.subj_dirs
    img_pth, seg_pth = dataset.get_paths()
    modalities = dataset.modalities
    transList = transforms
    outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
    call(f"mkdir {outpath}", shell=True)
    imgs = []
    masks = []
    # Define the list of helper functions for the transformation pipeline
    transform_pipeline = transforms.transforms_preproc()[1]
    for code, trans in transform_pipeline.items():
        if code in transList:
            apply_transforms(mask, code)
    for subj in subj_dirs:
        pth = os.path.join(data_dir, subj)
        for root, dirs, files in os.walk(pth):
            for fileName in files:
                if not fileName.endswith("-lbl.nii.gz") and not fileName.endswith("-stk.nii.gz"):
                    break
                proc_img = nib.load(os.path.join(root,fileName))
                proc_img = extract_imagedata(proc_img)
                proc_img_t = np.expand_dims(proc_img, axis=0)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_img_t = trans(proc_img_t)
                print("Label file: ", fileName, "shape is: ", proc_img_t.shape) 
                if fileName.endswith("-lbl.nii.gz"):
                    np.save(os.path.join(outpath, subj + "-lbl.npy"), proc_img_t)
                    masks.append(proc_img_t)

                elif fileName.endswith("-stk.nii.gz"): 
                    np.save(os.path.join(outpath, subj + "-stk.npy"), proc_img_t)
                    imgs.append(proc_img_t)
        
    return imgs, masks

def main():
    args = get_main_args()
    utils.set_cuda_devices(args)
    modalities = args.modal
    data_dir = args.data
    task = args.task
      
    print("Generating stacked nifti files.")
    startT = time.time()
    origData = MRIDataset(data_dir, task, modalities=modalities)
    prepare_nifty(origData)
    
    print("Loaded all nifti files and saved image data \nSaving a copy to images and labels folders")
    train = True if args.preproc_set == 'training' else False
    prepData = MRIDataset(data_dir, task, modalities=modalities)
    file_prep(prepData, args.data_grp, train)
    endT = time.time()
    print(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    print("Beginning Preprocessing.")
    startT2 = time.time()
    metadata = json.load(open(os.path.join(data_dir, "dataset.json"),"r"))
    transL = ['checkRAS', 'Znorm']
        # transform_pipeline = {
        # 'checkRAS' : to_ras,
        # 'CropOrPad' : crop_pad,
        # 'ohe' : one_hot_enc,
        # 'ZnormFore' : normalise_foreground,
        # 'MaskNorm' : masked,
        # 'Znorm': normalise}
    utils.run_parallel(preprocess_data(prepData, args, transList=[]), metadata, args)
    end2= time.time()
    print(f"Data Processing complete. Total time taken: {(end2 - startT2):.2f}")

if __name__=='__main__':
    main()