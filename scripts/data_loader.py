import torch
import os
import torch.utils.data as data_utils
from data_class import MRIDataset
from sklearn.model_selection import train_test_split
from data_transforms import define_transforms

def main():
    # FUNCTION JUST TO TEST DATA CLASS WORKS CORRECTLY

    data_dir = '/Users/alexandrasmith/Desktop/Workspace/Projects/UNN_BraTS23/data/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData/'
    data_folders = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not file == '.DS_Store']

    batch_size = 8

    dataloaders = load_data(data_folders, batch_size)

    # print(dataloaders)
    # training_set = dataloaders['train']

    # for img, label in training_set:
    #     print(img.shape)
    #     print(label.shape)
    
# MAIN FUNCTION TO USE
def load_data(data_folders, batch_size):
    '''
    Input:
    data_folders : list of all available data files
    batch_size (int) : define batch size to be used

    Returns dataloaders ready to be fed into model
    '''

    # Split data files
    train_files, val_files, test_files = split_data(data_folders, seed=42) # seed for reproducibiilty to get same split
    
    print(f"Number of training files: {len(train_files)}\nNumber of validation files: {len(val_files)}\nNumber of test: files {len(test_files)}")
    
    # Get data transforms
    data_transforms = define_transforms()

    image_datasets = {
    'train': MRIDataset(train_files, transform=data_transforms['train']),
    'val': MRIDataset(val_files,transform=data_transforms['val']),
    'test': MRIDataset(test_files, transform=data_transforms['test'])
    }
    # Create dataloaders
    # can set num_workers for running sub-processes
    dataloaders = {
        'train': data_utils.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True),
        'val': data_utils.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
        'test': data_utils.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)
    }

    return dataloaders

def split_data(data_folders, seed):
    '''
    Function to split dataset into train/val/test splits, given all avilable data.

    Returns:
    3 lists for each train, val, test sets, where each list contains the file names to be used in the set
    '''

    # Split into train (70), val (15), test (15) -- can edit
    train_files, test_files = train_test_split(data_folders, test_size=0.3, random_state=seed)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=seed)

    return train_files, val_files, test_files

if __name__=='__main__':
    main()