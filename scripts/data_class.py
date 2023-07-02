
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import nibabel as nib
import torchio
import utils
from utils import get_main_args

class MRIDataset(Dataset):
    """
    Given a set of images and corresponding labels (i.e will give it all training images + labels, and same for val and test)
    folder structure: subjectID/image.nii, seg.nii (i.e. contains 2 files)
    """

    def __init__(self, data_folders, transform=None, SSAtransform=None):
            self.data_folders = data_folders # path for each data folder in the set
            self.transform = transform
            self.SSAtransform = SSAtransform
            self.imgs = [] # store images to load (paths)
            self.lbls = [] # store corresponding labels (paths)
            # run through each subjectID folder
            for img_folder in data_folders:
                # check if current file is from SSA dataset
                self.SSA = True if 'SSA' in img_folder else False
                for file in os.list(img_folder):
                    # check folder contents
                    if os.path.isfile(os.path.join(img_folder, file)):
                        # Save segmentation mask (file path)
                        if file.endswith("-lbl.npy"):
                            self.lbls.append(os.path.join(img_folder, file))
                        elif file.endswith("-stk.npy")
                            # Save image (file path)
                            self.imgs.append(os.path.join(img_folder, file))

    def __len__(self):
        # Return the amount of images in this set
        return len(self.imgs)
    
    def __getitem__(self, idx):

        # Load files
        image = np.load(self.imgs[idx])
        mask = np.load(self.lbls[idx])

        # Convert to tensor
        image = torch.from_numpy(image) # 4, 240, 240, 155
        mask = torch.from_numpy(mask) # 240, 240, 155

        if self.transform is not None: # Apply general transformations
            # transforms such as crop, flip, rotate etc will be applied to both the image and the mask
            img = self.transform(img)
            mask = self.transform(mask)
        if self.SSA == False and self.SSAtransform is not None: # Apply transformation to GLI data to reduce quality (creating fake SSA data)
            # transforms such as blur, noise etc are NOT applied to mask as well
            img = self.SSAtransform(img)
        
        return img, mask
    
    def get_paths(self):
        return self.img_pth, self.seg_pth, self.proc_lbls, self.proc_imgs, self.imgs_npy, self.lbls_npy
    
    def get_subj_info(self):
        return self.subj_dir_pths, self.subj_dirs
        #, self.SSA
    
    def get_transforms(self):
        return self.transform


############ THIS SECTION WILL BE REMOVED ##########
"""
AA __init__ : delete below once copied to local system

    def __init__(self, subj_dirL, task=args.task, modalities=args.modal, transform=None, SSAtransform=None):
        self.data_dir = subj_dirL # path for each subject folder in the set
        self.modalities = modalities
        self.task = task

        self.transform = transform
        self.SSAtransform = SSAtransform
        
        self.subj_dirs, self.subjIDls, self.subj_dir_pths = [],[],[]
        # store images to load (paths)
        self.img_pth, self.proc_imgs, self.imgs_npy = [],[],[]
        # store corresponding labels (paths)
        self.seg_pth, self.proc_lbls, self.lbls_npy = [],[],[]

        file_ext_dict_prep = {
            "-seg.nii.gz": self.seg_pth,
            "-lbl.nii.gz": self.proc_lbls,
            "-stk.nii.gz": self.proc_imgs,
            **{f"-{m}.nii.gz": self.img_pth for m in modalities},
        }

        file_ext_dict_aug = {
            "-lbl.npy": self.imgs_npy,
            "-stk.npy": self.lbls_npy
        }

        for root, dirs, files in os.walk(self.data_dir):
            for directory in sorted(dirs, key=lambda x: x.lower(), reverse=True):
                if not "BraTS-" in directory:
                    break
                else:
                    self.subj_dirs.append(str(directory))
                    self.subj_dir_pths.append(os.path.join(root,directory))
                    #self.subjIDls.append(self.subj_dirs)
                    self.SSA = 'SSA' in self.subj_dirs
            for file in files:
                file_pth = os.path.join(root, file)
                if os.path.isfile(file_pth) and task=='data_prep':
                    for ext, list_to_append in file_ext_dict_prep.items():
                        if file.endswith(ext):
                            #print(file_pth)
                            list_to_append.append(file_pth)
                else:
                    for ext, list_to_append in file_ext_dict_aug.items():
                        if file.endswith(ext):
                            #print(file_pth)
                            list_to_append.append(file_pth)
"""