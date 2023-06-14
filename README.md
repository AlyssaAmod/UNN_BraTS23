# UNN_BraTS23
MICCAI 2023 Brain Tumour Segmentation Challenge

Project Structure
------------

    ├── README.md                  <- The top-level README for developers using this project.
    ├── notebooks
    │   └── data-exploration       <- perform initial exploratory data analysis
    │ 
    ├── scripts
    │   ├── data_preparation       <- read in dataset, extract header info, set up labels e.t.c. custom Dataset class
    │   ├── data_preprocessing     <- crop unnecessary background, ...
    │   └── data_loader            <- split dataset into train/val/test, process into PyTorch dataloaders, data ready to be fed into model
    │  
    ├── reports                   <- contains all generated graphics for reporting

------------
