import torch
import os
import torch.utils.data as data_utils
import json
from subprocess import call
from data_class import MRIDataset
from sklearn.model_selection import train_test_split
from data_transforms import define_transforms
from utils.utils import get_main_args
import pickle
import glob

def main():
    # FUNCTION JUST TO TEST DATA CLASS WORKS CORRECTLY
    args = get_main_args()

    ## Alex: testing from terminal
    data_dir = '/scratch/guest187/Data/train_all'
    data_folders = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not (file.endswith(".json") or file == 'images' or file == 'labels' or file == 'ATr_prepoc')]
    batch_size = 8

    # print(data_folders)

    # Code to save config file
    # pickle.dump(
    #     {
    #         "patch_size": [128, 128, 128],
    #         "spacings": [1.0, 1.0, 1.0],
    #         "n_class": 4,
    #         "in_channels": 4,
    #     },
    #     open(os.path.join('/scratch/guest187/Data/train_all', "config.pkl"), "wb"),
    # )

    dataloaders = load_data(data_dir, batch_size, args)
    print(dataloaders)
    training_set = dataloaders['train']
    
    # for img, label in training_set:
    #     print(f"Image shape: {img.shape}")
    #     print(f"Label shape: {label.shape}")
    
# MAIN FUNCTION TO USE
def load_data(args, data_transforms):
    '''
    Input:
    data_folders : list of all available data files
    batch_size (int) : define batch size to be used

    Returns dataloaders ready to be fed into model
    '''
    if args.seed != None:
        seed=args.seed
    else:
        seed=None

    if args.data_used == 'all':
        data_folders = glob.glob(os.path.join(args.data, "BraTS*"))
    elif args.data_used == "GLI":
        data_folders = [folder for folder in os.listdir(args.data) if 'GLI' in folder]
    elif args.data_used == 'SSA':
        data_folders = [folder for folder in os.listdir(args.data) if 'SSA' in folder]

    # Split data files
    train_files, val_files = split_data(data_folders, seed) # seed for reproducibiilty to get same split
    
    print(f"Number of training files: {len(train_files)}\nNumber of validation files: {len(val_files)}")
    
    # Get data transforms
    # data_transforms = define_transforms(n_channels)
    # fakeSSA transforms are applied to GLI data to worse their image quality
    # image_datasets = {
    # 'train': MRIDataset(train_files, transform=data_transforms['train'], SSAtransform=data_transforms['fakeSSA']),

    image_datasets = {
    'train': MRIDataset(args.data, train_files, transform=data_transforms['train']),
    'val': MRIDataset(args.data, val_files, transform=data_transforms['val']),
    # 'test': MRIDataset(args, test_files, transform=data_transforms['test'])
    }

    # Create dataloaders
    # can set num_workers for running sub-processes
    dataloaders = {
        'train': data_utils.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, drop_last=True),
        'val': data_utils.DataLoader(image_datasets['val'], batch_size=args.val_batch_size, shuffle=True),
        # 'test': data_utils.DataLoader(image_datasets['test'], batch_size=args.val_batch_size, shuffle=True)
    }

    # Save data split
    splitData = {
        'subjsTr' : train_files,
        'subjsVal' : val_files,
        # 'subjsTest' : test_files    
    }
    with open(args.data + str(args.data_used) + ".json", "w") as file:
        json.dump(splitData, file)

    return dataloaders

def split_data(data_folders, seed):
    '''
    Function to split dataset into train/val/test splits, given all avilable data.

    Returns:
    3 lists for each train, val, test sets, where each list contains the file names to be used in the set
    '''

    # Split into train (70), val (15), test (15) -- can edit
    train_files, val_files = train_test_split(data_folders, test_size=0.7, random_state=seed)
    # val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=seed)

    return train_files, val_files

if __name__=='__main__':
    main()