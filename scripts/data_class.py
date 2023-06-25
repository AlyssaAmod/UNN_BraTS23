
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import nibabel as nib
import torchio
import utils
from utils import get_main_args

args = get_main_args()
class MRIDataset(Dataset):
    # Given a set of images and corresponding labels (i.e will give it all training images + labels, and same for val and test)
    # folder structure: subjectID/image.nii, seg.nii (i.e. contains 2 files)
    def __init__(self, data_dir, task, modalities=[], transform=None, SSAtransform=None):
        self.data_dir = data_dir # path for each data folder in the set
        self.modalities = modalities
        self.task = task

        self.transform = transform
        self.SSAtransform = SSAtransform
        
        subj_dirs = sorted(os.listdir(self.data_dir))

        img_pth = []
        seg_pth = []
        proc_imgs = []
        proc_lbls = []
        
        self.imgs = [] # store images to load (paths)
        self.lbls = [] # store corresponding labels (paths)
        
        
        if task == "data_prep":
            for subj in sorted(os.listdir(data_dir)):
                # run through each subjectID folder
                subj_dir = os.path.join(data_dir, subj)
                self.subj_dir = subj_dir
                subjID = str(subj_dir)
                self.subjID = subjID                
                self.SSA = True if 'SSA' in subjID else False
                for file in os.listdir(subj_dir):
                    # check folder contents
                    if os.path.isfile(os.path.join(subj_dir, file)):
                        # Save original segmentation mask (file path)
                        if file.endswith("-seg.nii.gz"):
                            seg_pth.append(os.path.join(subj_dir, file))
                            self.seg_pth = seg_pth
                        elif [file.endswith(f"-{m}.nii.gz") for m in modalities]:
                            # Save original image (file path)
                            img_pth.append(os.path.join(subj_dir, file))
                            self.img_pth = img_pth                            # self.img_pth.append([(os.path.join(self.subj_dir, self.subjID + f"-{m}.nii.gz")) for m in modalities])
                        # # Save pre-processed segmentation mask (file path)
                        # if file.endswith("-lbl.nii.gz"):
                        #     proc_lbls.append(os.path.join(subj_dir, file))
                        #     self.proc_lbls = proc_lbls
                        # if file.endswith("-stk.nii.gz"):
                        #     # Save preprocessed image (file path)
                        #     proc_imgs.append(os.path.join(subj_dir, file))
                        #     self.proc_imgs = proc_imgs
        else: 
            for img_folder in data_dir:
                # check if current file is from SSA dataset
                self.SSA = True if 'SSA' in img_folder else False
                for file in os.list(img_folder):
                    # check folder contents
                    if os.path.isfile(os.path.join(img_folder, file)):
                        # Save segmentation mask (file path)
                        if file.endswith("-lbl.npy"):
                            self.lbls.append(os.path.join(img_folder, file))
                        elif file.endswith("-stk.npy"):
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
        return self.img_pth, self.seg_pth #self.proc_lbls, self.proc_imgs
    
    def get_subj_info(self):
        self.subj_dir, self.subjID#, self.SSA
    
    def get_transforms(self):
        self.transform


############# OLD CODE -- DEL #########

        # # run through each subjectID folder
        # for img_folder in data_dir:
        #     # check if current file is from SSA dataset
        #     self.SSA = True if 'SSA' in img_folder else False
        #     for file in os.list(img_folder):
        #         # check folder contents
        #         if os.path.isfile(os.path.join(img_folder, file)):
        #             # Save segmentation mask (file path)
        #             if file.endswith("-seg.nii.gz"):
        #                 self.lbls.append(os.path.join(img_folder, file))
        #             else:
        #                 # Save image (file path)
        #                 self.imgs.append(os.path.join(img_folder, file))