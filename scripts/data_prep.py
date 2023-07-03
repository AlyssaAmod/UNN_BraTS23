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
import nibabel as nib
import numpy as np
import torch
import torchio as tio

import utils
from utils import get_main_args
from utils import extract_imagedata
from data_transforms import transforms_preproc

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def data_preparation(data_dir, args):
    
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
        NifTI files:.
            subjIDxxx-stk.nii.gz == stacked nifti img data 
            subjIDxxx-lbl.nii.gz == seg mask img data

    """
    
    # Step 1: Initialisation

    #data_dir = args.data # path for each subject folder in the set
    modalities = args.modal
    subj_dirs, subj_dir_pths = [],[]
    # store images to load (paths)
    img_pth, seg_pth = [],[]
    file_ext_dict_prep = {
        **{f"-{m}.nii.gz": img_pth for m in modalities},
        "-seg.nii.gz": seg_pth}

    #Loop through main data folder to generate lists of paths
    print(f"Generating dataset paths from data folder: {data_dir}")
    for root, dirs, files in os.walk(data_dir):
        for directory in sorted(dirs, key=lambda x: x.lower(), reverse=False):
            if not "BraTS-" in directory:
                break
            else:
                subj_dirs.append(str(directory))
                subj_dir_pths.append(os.path.join(root,directory))
        for file in files:
            file_pth = os.path.join(root, file)
            if os.path.isfile(file_pth) and args.task=='data_prep':
                print(os.path.dirname(file_pth))
                for ext, list_to_append in file_ext_dict_prep.items():
                    if file.endswith(ext):
                        # print(file_pth)
                        list_to_append.append(file_pth)        
    print("Total Number of Subjects is: ", len(subj_dirs))
    for k,v in file_ext_dict_prep.items():
        file_ext_dict_prep[k] = sorted(v, key=lambda x: x.lower())
    print(f"Saving path lists to file: {args.preproc_set}_paths.json")
    with open(os.path.join(data_dir, f'{args.preproc_set}_paths.json'), 'w') as file:
        json.dump(file_ext_dict_prep, file)
    del file_ext_dict_prep
    # Step 2: Stack modalities into 1 nii file, and extract header information
    print("Preparing stacked nifty files")

    img_shapes = {}
    res = {}
    img_modality = []
    ext_dict_modal = {**{f"-{m}.nii.gz": img_modality for m in modalities}}
    # store paths
    proc_imgs, proc_lbls = [],[]
    file_ext_dict_prep2 = {
        "-stk.nii.gz": proc_imgs,
        "-lbl.nii.gz": proc_lbls
    }
    
    for sub_dir in sorted(subj_dir_pths, key=lambda x: x.lower(), reverse=False):
        if not "BraTS-" in sub_dir:
            break
        subj_id = os.path.basename(sub_dir)
        print("Working on subj: ", subj_id)
        
    #Load nifti file for each scanning sequence
        print("Loading and stacking modalities")
        img_paths = [s for s in img_pth if subj_id in s]
        loaded_modalities = [nib.load(path) for path in img_paths]
        t1n, t1c, t2w, t2f = loaded_modalities
        img_modality.extend([t1n, t1c, t2w, t2f]) 
        affine, header = t2f.affine, t2f.header
        
        res[f'{subj_id}_RES']=header.get_zooms()
    
    #Stack all into one nifti file
        imgs = np.stack([extract_imagedata(modality) for modality in loaded_modalities], axis=-1)
        shapes = {modality: imgs[..., i].shape for i, modality in enumerate(modalities)}
        img_shapes[f'{subj_id}'] = shapes
        print("Image shapes: ", img_shapes[f'{subj_id}'])
        imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
        nib.save(imgs, os.path.join(sub_dir, subj_id + "-stk.nii.gz"))
        proc_imgs.append(os.path.join(sub_dir, subj_id + "-stk.nii.gz"))
        del imgs
        del shapes
        del img_modality            
        
    # Step 3: Load and save seg
        print("Loading and saving segmentation")
        seg = nib.load(os.path.join(sub_dir, subj_id + "-seg.nii.gz"))
        seg_affine, seg_header = seg.affine, seg.header
        seg = extract_imagedata(seg, "unit8")
        #seg[vol == 4] = 3 --> not sure what this does yet
        seg = nib.nifti1.Nifti1Image(seg, seg_affine, header=seg_header)
        print("Seg Shape", seg.shape)
        nib.save(seg, os.path.join(sub_dir, subj_id + "-lbl.nii.gz"))
        proc_lbls.append(os.path.join(sub_dir, subj_id + "-lbl.nii.gz"))
        del seg               
    # save a few bits of info into a json 
    
    with open(os.path.join(data_dir, f'{args.preproc_set}_pathsSTK.json'), 'w') as file:
        json.dump(file_ext_dict_prep2, file)
    
    print("Saving shape & resolution data per subject")
    img_info = {
        "img_shapes": img_shapes,
        "res": res,
        "all_paths": img_pth
        }
    with open(os.path.join(data_dir, 'img_info.json'), 'w') as file:
        json.dump(img_info, os.path.join(data_dir,file), cls=NumpyEncoder)
    del img_shapes
    del res
    del img_pth
    return 


def file_prep(data_dir, dataMode, args):
    """ 
    This an extra function to save a copy of the image data extracted from each volume.
    data_loader and trainer do not require these data as they are stored in the original subject folders as well.

    Creates a json file with
        A dictionary of dummy coding for seg labels as provided by BraTS
            "labels" : {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
        A dictionary of dummy coding for each modality
            "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
        A dictionary of dictionaries containing the image-label path pairs
            "training": [{"image": "images/subjIDxxx.nii.gz", "label": "labels/subjIDxxx_seg.nii.gz"}
    """

    modalities = args.modal
    filePaths = json.load(open(data_dir,f'{args.exec_mode}_paths2.json', "r"))
    subjInfo = json.load(open(data_dir,f'subj_info.json', "r"))

    stk = sorted(filePaths["-stk.nii.gz"], key=lambda x: x.lower(), reverse=False)
    lbl = sorted(filePaths["-lbl.nii.gz"], key=lambda x: x.lower(), reverse=False)
    subj_dirs = subjInfo["subj_dirs"]
    subj_id = subjInfo["subjIDs"]

    print("Saving subject folder paths and list of IDs. Total subjects is: ", len(subj_dirs))    
    subj_info = {
        "nSubjs" : len(subj_dirs),
        "subjIDs" : subj_dirs,
        "subj_dirs" : subj_dir_pths
    }
    with open(os.path.join(data_dir, "subj_info.json"), "w") as file:
        json.dump(subj_info,file)
    
    stk_path, lbls_path = os.path.join(data_dir, f"images_orig-{dataMode}"), os.path.join(data_dir, f"labels_orig-{dataMode}")
    call(f"mkdir -p {stk_path}", shell=True)
    if args.preproc_set != "test":
        call(f"mkdir -p {lbls_path}", shell=True)
    
    imagesF, labelsF = [], []
    file_ext_dict2 = {
        "-lbl.nii.gz": labelsF,
        "-stk.nii.gz": imagesF
    }

    for dir in sorted(subj_dirs, key=lambda x: x.lower(), reverse=False):
        if not "BraTS-" in dir:
            break
        id_check = os.path.basename(dir)
        for i in range(len(subj_id)):
            if id_check == os.path.dirname(lbl[i]):
                lbl_file = os.path.basename(lbl[i]) 
                labelsF.append(os.path.join(dir, lbl_file))
                call(f"cp {lbl[i]} {lbls_path}", shell=True)
            if id_check == os.path.dirname(stk[i]):            
                stk_file = os.path.basename(stk[i]) 
                imagesF.append(os.path.join(dir, stk_file))
                call(f"cp {stk[i]} {stk_path}", shell=True)

    if args.preproc_set == "training":
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
    if args.preproc_set == "training":
        key = "training"
        data_pairs_fold = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs_fold = [{"image": img} for img in images]

    # sAve some json files for dataloading
    dataset = {
        "labels": labels_dict,
        "modality": modality,
        "subjIDs" : subj_id,
        key: data_pairs}
    with open(os.path.join(data_dir, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

    datasetFold = {
        "labels": labels_dict,
        "modality": modality,
        "subjIDs" : subj_id,
        key: data_pairs_fold}
    with open(os.path.join(data_dir, "datasetFold.json"), "w") as outfile:
        json.dump(datasetFold, outfile)

    del data_pairs_fold
    del dataset
    del imagesF
    del labelsF

def preprocess_data(data_dir, args, transList):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
    '''
    # img is still a list of arrays of the 4 modalities from data files
    # mask is 3d array

    # return img as list of arrays, and mask as before

    filePaths = json.load(open(data_dir,f'{args.preproc_set}_paths.json', "r"))
    subjInfo = json.load(open(data_dir,f'subj_info.json', "r"))

    stk = sorted(filePaths["-stk.nii.gz"], key=lambda x: x.lower(), reverse=False)
    lbl = sorted(filePaths["-lbl.nii.gz"], key=lambda x: x.lower(), reverse=False)
    subj_dirs = subjInfo["subj_dirs"]
    subj_id = subjInfo["subjIDs"]

    outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
    call(f"mkdir -p {outpath}", shell=True)
    imgs_npy = []
    lbls_npy = []

    imgs = []
    masks = []
    # Define the list of helper functions for the transformation pipeline
    transform_pipeline = transforms_preproc()[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dir in sorted(subj_dirs, key=lambda x: x.lower(), reverse=False):
        if not "BraTS-" in dir:
            break
        id_check = os.path.basename(dir)
        for i in range(len(subj_id)):
            if id_check == os.path.dirname(lbl[i]):
                proc_lbl = nib.load(lbl[i])
                proc_lbl = extract_imagedata(proc_lbl)
                proc_lbl_t = (torch.from_numpy(proc_lbl)).to(device)
                proc_lbl_t = torch.unsqueeze(proc_lbl_t, axis=0)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_lbl_t = trans(proc_lbl_t)
                np.save(os.path.join(dir, id_check + "-lbl.npy"), proc_lbl_t)
                lbls_npy.append(os.path.join(dir, id_check + "-lbl.npy"))
                masks.append(proc_lbl_t)
            del proc_lbl
            del proc_lbl_t
            if id_check == os.path.dirname(stk[i]):            
                proc_img = nib.load(stk[i])
                proc_img = extract_imagedata(proc_img)
                proc_img_t = (torch.from_numpy(proc_img)).to(device)
                # proc_img_t = np.expand_dims(proc_img, axis=0)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_img_t = trans(proc_img_t)
                np.save(os.path.join(dir, id_check + "-stk.npy"), proc_img_t)
                imgs_npy.append(os.path.join(dir, id_check + "-stk.npy"))
                imgs.append(proc_img_t)
            del proc_img
            del proc_img_t
    
        
    datasetNPY = {
        "img_folders" : subj_dirs,
        "img_np_pth" : imgs_npy,
        "mask_np_pth" : lbls_npy,
        "npy_pairPths" : [{"image": img, "label": lbl} for (img, lbl) in zip(imgs_npy, lbls_npy)]
    }
    with open(os.path.join(data_dir, "dataset.json"), "a") as outfile:
        json.dump(datasetNPY, outfile)


def main():
    args = get_main_args()
    utils.set_cuda_devices(args)
    data_dir = args.data
    
    print("Generating stacked nifti files.")
    
    startT = time.time()
    utils.run_parallel(data_preparation(data_dir, args),[data_dir, args])

    data_preparation(data_dir, args)
    
    print("Loaded all nifti files and saved image data")
    print("Saving a copy to images and labels folders")
    
    train = args.preproc_set
    file_prep(data_dir, args.data_grp, train)
    endT = time.time()
    
    print(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    print("Beginning Preprocessing.")
    startT2 = time.time()
    # metadata = json.load(open(os.path.join(data_dir, "dataset.json"),"r"))
    transL = ['checkRAS', 'Znorm']
        # OPTIONS ARE:
        # 'checkRAS' : to_ras,
        # 'CropOrPad' : crop_pad,
        # 'ohe' : one_hot_enc,
        # 'ZnormFore' : normalise_foreground,
        # 'MaskNorm' : masked,
        # 'Znorm': normalise
    utils.run_parallel(preprocess_data(data_dir, args, transL),[data_dir, args, transL])
    # preprocess_data(data_dir, args, transList=transL)
    end2= time.time()
    print(f"Data Processing complete. Total time taken: {(end2 - startT2):.2f}")

if __name__=='__main__':
    main()