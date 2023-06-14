import torch
import os
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import nibabel as nib
from data_preprocessing import preprocess_data # needs to be created

class MRIDataset(Dataset):
    # Given a set of images and corresponding labels (i.e will give it all training images + labels, and same for val and test)
    def __init__(self, data_folders, transform=None):
        self.data_folders = data_folders # path for each data folder in the set

        self.imgs = {} # store images
        self.lbls = {} # store corresponding labels

        # Load the images and corresponding labels
        for i, img_folder in enumerate(data_folders):
            print(img_folder)
            modalities = []
            for file in os.listdir(img_folder):
                # Check folder contents
                if all(substring not in file for substring in ["t1c", "t1n", "t2f", "t2w", "seg"]):
                    raise Exception(f"File found that is not an imaging modality or ground truth. /n File name: {file}")
                # Save segmentation mask
                if file.endswith("-seg.nii.gz"):
                    self.lbls.append(nib.load(os.path.join(img_folder, file)))
                else:
                    # Save image modalities to list
                    modalities.append(nib.load(os.path.join(img_folder, file)))
            # Save images containing all modalities
            self.imgs.append(modalities)
            # mask = nib.load(os.path.join(data_dir + img_folder, img_folder + "-seg.nii.gz"))
        
            # img_volumes = [nib.load(os.path.join(data_dir + img_folder, img_folder + f"-{m}.nii.gz")) for m in ["t1c", "t1n", "t2f", "t2w"]]

            # apply preprocessing
            # from data_preprocessing.py have one function that applies all preprocessing we want
            # same preprocessing applied to data and masks
            self.imgs = [preprocess_data(v) for v in self.imgs]
            self.lbls = [preprocess_data(v) for v in self.lbls]

    def __len__(self):
        # Return the amount of images in this set
        return len(self.imgs)
    
    def __getitem__(self):

        if self.transform is not None: # Apply transformations
            img, mask = self.transform((img, mask))

        # Convert to tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask

data_folders = os.listdir('/Users/alexandrasmith/Desktop/Workspace/Projects/UNN_BraTS23/data/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData/')
print(data_folders)

print(MRIDataset(data_folders))