import torch
import os
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import nibabel as nib
import torchio
from utils import get_main_args

args = get_main_args()
class MRIDataset(Dataset):
    # Given a set of images and corresponding labels (i.e will give it all training images + labels, and same for val and test)
    # folder structure: subjectID/image.nii, seg.nii (i.e. contains 2 files)
    def __init__(self, data_dir, modalities=[], transform=None, SSAtransform=None):
        self.data_dir = data_dir # path for each data folder in the set
        self.modalities = modalities

        self.transform = transform
        self.SSAtransform = SSAtransform
        
        subjID = sorted(os.listdir(self.data_dir))
        self.subjID = subjID
        self.SSA = True if 'SSA' in subjID else False

        subj_dir = os.path.join(data_dir, self.subjID)
        self.subj_dir = subj_dir
        self.img_pth = []
        self.seg_pth = []
        
        self.imgs = [] # store images to load (paths)
        self.lbls = [] # store corresponding labels (paths)
        
        # run through each subjectID folder
        for file in subj_dir:
            # check folder contents
            if os.path.isfile(os.path.join(subj_dir, file)):
                # Save original segmentation mask (file path)
                if file.endswith("-seg.nii.gz"):
                    self.seg_pth(os.path.join(subj_dir, file))
                elif [file.endswith(f"-{m}.nii.gz") for m in modalities]:
                    # Save original image (file path)
                    self.img_pth.append(os.path.join(subj_dir, file))
                    # self.img_pth.append([(os.path.join(self.subj_dir, self.subjID + f"-{m}.nii.gz")) for m in modalities])
                # Save pre-processed segmentation mask (file path)
                if file.endswith("-lbl.nii.gz"):
                    self.lbls.append(os.path.join(subj_dir, file))
                else:
                    # Save preprocessed image (file path)
                    self.imgs.append(os.path.join(subj_dir, file))


    def __len__(self):
        # Return the amount of images in this set
        return len(self.imgs)
    
    def __getitem__(self, idx):

        # Load files
        image = nib.load(self.imgs[idx]).get_fdata()
        mask = nib.load(self.lbls[idx]).get_fdata()

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