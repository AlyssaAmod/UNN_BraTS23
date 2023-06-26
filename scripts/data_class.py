
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
        
        self.subj_dirs = sorted(os.listdir(data_dir))
        self.subjIDls = []
        self.img_pth = []
        self.seg_pth = []
        self.proc_imgs = []
        self.proc_lbls = []
        self.subj_dir_pths = []

        self.imgs = [] # store images to load (paths)
        self.lbls = [] # store corresponding labels (paths)

        if task == "data_prep":

            for root, dirs, files in os.walk(self.data_dir):
                for dir in dirs:
                    self.subj_dir_pths.append(os.path.join(root,dir))
                    self.subjID = str(dir)
                    self.subjIDls.append(self.subjID)           
                    self.SSA = True if 'SSA' in self.subjID else False
                for file in files:
                    # print(file)
                # check folder contents
                    file_pth = os.path.join(root, file)
                    # print(file_pth)
                    if os.path.isfile(file_pth):
                        # Save original segmentation mask (file path)
                        if file.endswith("-seg.nii.gz"):
                            self.seg_pth.append(file_pth)
                        elif [file.endswith(f"-{m}.nii.gz") for m in modalities]:
                            # Save original image (file path)
                            self.img_pth.append(file_pth)
                        # Save pre-processed segmentation mask (file path)
                        elif file.endswith("-lbl.nii.gz"):
                            self.proc_lbls.append(file_pth)
                        elif file.endswith("-stk.nii.gz"):
                            # Save preprocessed image (file path)
                            self.proc_imgs.append(file_pth)
        else: 
            for img_folder in data_dir:
                # check if current file is from SSA dataset
                self.SSA = True if 'SSA' in img_folder else False
                for file in os.listdir(img_folder):
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