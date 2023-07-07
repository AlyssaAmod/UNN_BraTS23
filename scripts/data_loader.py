import torch
import os
import torch.utils.data as data_utils
import json
from subprocess import call

from data_class import MRIDataset
from sklearn.model_selection import train_test_split
from data_transforms import define_transforms
from utils.utils import get_main_args

def main():
    # FUNCTION JUST TO TEST DATA CLASS WORKS CORRECTLY
    args = get_main_args()
    # utils.set_cuda_devices(args)
    data_dir = args.data
    outpath = os.path.join(args.data_dir, args.data_grp + "_trainingSplits")
    call(f"mkdir -p {outpath}", shell=True)

    # data_dir = '/scratch/guest187/BraTS_Africa_data/Baseline/NewScripts_SamplesTest/Samples'
    # data_folders = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not file == '.DS_Store']
    
    # Use for testing data loaders until dataset.json is available
    # QUESTION: what is the if not any statement checking for if this is meant to create file paths to subject folders?
    data_folders = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not any(i in file for i in ['stk', 'lbl']) and file.startswith('BraTS-')]

    # datasetInfo = json.load(open(os.path.join(data_dir,"dataset.json"), "r"))

    # print(datasetInfo)
    
    # data_folders = datasetInfo["img_folders"]
    # img_lbl_npy = datasetInfo["npy_pairPths"]
    # img_np_pth = datasetInfo["img_np_pth"]
    # mask_np_pth = datasetInfo["mask_np_pth"]

    print(data_folders)

    batch_size = args.batch_size

    ## Alex: testing from terminal
    # data_dir = data_dir='/scratch/guest187/Data/train_all'
    # data_folders = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not (file.endswith(".json") or file == 'images' or file == 'labels' or file == 'ATr_prepoc')]
    # batch_size = 16

    # print(data_folders)

    dataloaders = load_data(data_folders, batch_size, args)
    print(dataloaders)
    training_set = dataloaders['train']
    
    for img, label in training_set:
        print(img.shape)
        print(label.shape)
    
# MAIN FUNCTION TO USE
def load_data(data_dir, batch_size, args):
    '''
    Input:
    data_folders : list of all available data files
    batch_size (int) : define batch size to be used

    Returns dataloaders ready to be fed into model
    '''

    data_folders = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not (file.endswith(".json") or file == 'images' or file == 'labels' or file == 'ATr_prepoc')]

    # Split data files
    train_files, val_files, test_files = split_data(data_folders, seed=42) # seed for reproducibiilty to get same split
    
    print(f"Number of training files: {len(train_files)}\nNumber of validation files: {len(val_files)}\nNumber of test: files {len(test_files)}")
    
    # Get data transforms
    data_transforms = define_transforms()
    # fakeSSA transforms are applied to GLI data to worse their image quality
    # image_datasets = {
    # 'train': MRIDataset(train_files, transform=data_transforms['train'], SSAtransform=data_transforms['fakeSSA']),
    # 'val': MRIDataset(val_files, transform=data_transforms['val']),
    # 'test': MRIDataset(test_files, transform=data_transforms['test'])
    # }
    image_datasets = {
    'train': MRIDataset(train_files, transform=data_transforms['train']),
    'val': MRIDataset(val_files, transform=data_transforms['val']),
    'test': MRIDataset(test_files, transform=data_transforms['test'])
    }

    # splitData = {
    #     'subjsTr' : image_datasets['train'].subj_dirs,
    #     'subjsVal' : image_datasets['val'].subj_dirs,
    #     'subjsTest' : image_datasets['test'].subj_dirs    
    # }

    splitData = {
        'subjsTr' : train_files,
        'subjsVal' : val_files,
        'subjsTest' : test_files    
    }
    
    # with open(os.path.join(outpath, "trainSplit.json"), "a") as outfile:
    #     json.dump(splitData, outfile)

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