# UNN_BraTS23
MICCAI 2023 Brain Tumour Segmentation Challenge

Project Organisation
------------

  ├── README.md          <- The top-level README for developers using this project.
  ├── ##notebooks##
  │   └── data-exploration       <- perform initial exploratory data analysis
  │ 
  ├── ##scripts##
  │   ├── data_preparing         <- read in dataset, extract header info, set up labels e.t.c.
  │   ├── data_preprocessing     <- crop unnecessary background, ...
  │   └── data_loader            <- split dataset into train/val/test, process into PyTorch dataloaders, data ready to be fed into model
  │  
  ├── ##reports##                    <- contains all generated graphics for reporting

------------


    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
