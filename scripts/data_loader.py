from torchvision import transforms
import torch.utils.data as data_utils
from data_preparation import MRIDataset
from sklearn.model_selection import train_test_split

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

#! TO DO: we must fill in the transforms we want to apply
def define_transforms():
    # Initialise data transforms
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(INPUT_SIZE),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
        ]),
        'val': transforms.Compose([
            # transforms.Resize(INPUT_SIZE),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
        ]),
        'test' : transforms.Compose([
            # transforms.Resize(INPUT_SIZE),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
        ])
    }

    return data_transforms

def split_data(data_folders, splt: list, seed):
    '''
    Function to split dataset into train/val/test splits, given all avilable data.

    Returns:
    3 lists for each train, val, test sets, where each list contains the file names to be used in the set
    '''

    # Split into train (70), val (15), test (15) -- can edit
    train_files, test_files = train_test_split(data_folders, test_size=0.3, random_state=seed)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=seed)

    return train_files, val_files, test_files

