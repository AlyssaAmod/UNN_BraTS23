import torch
import os
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import nibabel as nib
from data_preprocessing import preprocess_data

class MRIDataset(Dataset):
    # Given a set of images and corresponding labels (i.e will give it all training images + labels, and same for val and test)
    def __init__(self, data_folders, transform=None):
        self.data_folders = data_folders # path for each data folder in the set

        self.imgs = {} # store images to load (paths)
        self.lbls = {} # store corresponding labels (paths)

        # Load the images and corresponding labels
        for i, img_folder in enumerate(data_folders):
            print(img_folder)
            modalities = []
            for file in os.listdir(img_folder):
                # Check folder contents
                if all(substring not in file for substring in ["t1c", "t1n", "t2f", "t2w", "seg"]):
                    raise Exception(f"File found that is not an imaging modality or ground truth. /n File name: {file}")
                if os.path.isfile(os.path.join(img_folder, file)):
                    # Save segmentation mask (file paths)
                    if file.endswith("-seg.nii.gz"):
                        self.lbls.append(os.path.join(img_folder, file))
                    else:
                        # Save image modalities to list (file paths)
                        modalities.append(os.path.join(img_folder, file))
            # Save images containing all modalities
            self.imgs.append(modalities)

            # apply preprocessing
            self.imgs, self.lbls = [preprocess_data(v) for v in enumerate(self.imgs, self.lbls)]

    def __len__(self):
        # Return the amount of images in this set
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # Loads and returns sample

        # Load files
        data = [nib.load(nib.load(img_path) for img_path in self.imgs[idx])] # list of modalities
        tnsrs = [torch.from_numpy(mod) for mod in data] # convert to tensors
        # concatenate modalitites to make tensor
        img = torch.stack(tnsrs) # 4, 240, 240, 155
        print(img.shape)

        mask = nib.load(self.lbls[idx])
        mask = torch.from_numpy(mask) # convert to tensor

        if self.transform is not None: # Apply transformations
            img, mask = self.transform((img, mask))
        
        return img, mask

data_folders = os.listdir('/Users/alexandrasmith/Desktop/Workspace/Projects/UNN_BraTS23/data/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData/')

MRIDataset(data_folders)