    # UNN_BraTS23
MICCAI 2023 Brain Tumour Segmentation Challenge
## Running these scripts

1. data_prep : is an independent script, meant to only be run once to preprocess (and save) all raw data provided from the challenge, so that the data is ready to be put into dataloaders

2. data_loader : reads from data_class and data_transforms, gets the data ready for model (will be read in from training script)

3. training script: The training and validation data from the data_loader are passed to the trainer, respectively. 

## Project Structure
------------
├── Adewole et al_The Brain Tumor Segmentation (BraTS) Challenge 2023.pdf
├── Background.md
├── LICENSE
├── notebooks
│   ├── data-exploration.ipynb
│   ├── test_scripts.ipynb
│   ├── transforms-exploration.ipynb
│   └── usefulFx.ipynb
├── playground
│   ├── data_prep-augs
│   │   ├── data_prepFakeSSA.py
│   │   ├── prepoc_augs
│   │   │   ├── Augs.py
│   │   │   └── stdPreproc.py
│   │   └── UNN_datapreparation_v0.py
│   ├── modelZoo_playground
│   │   ├── model.py
│   │   ├── nnunet
│   │   │   ├── brats22_model.py
│   │   │   ├── loss.py
│   │   │   ├── metrics.py
│   │   │   ├── nn_unet.py
│   │   │   └── nnunet_trainer.py
│   │   ├── optimized_U-net.py
│   │   └── training.py
│   ├── training_fn_playground.py
│   └── visualisations.py
├── README.md
├── reports
│   ├── BraTS-SSA-00008-000.png
│   ├── BraTS-SSA-00008-000_seg.png
│   ├── BraTS-SSA-00008-000_t1c.png
│   ├── BraTS-SSA-00008-000_t1n.png
│   ├── BraTS-SSA-00008-000_t2f.png
│   ├── BraTS-SSA-00008-000_t2w.png
│   ├── t1c.png
│   ├── t1n.png
│   ├── t2f.png
│   └── transforms.py
├── results
│   └── optiNet_basline.md
└── scripts
    ├── data_class.py
    ├── data_loader.py
    ├── data_preparation
    │   ├── data_preparation_OptiNet.py
    │   └── data_preparation_UNN.py
    ├── data_transforms.py
    ├── modelZoo_monai.py
    ├── monai_trainer.py
    ├── MONAI_Unet_Brats.ipynb
    ├── salloc_InteractiveJob.sh
    └── utils
        ├── logger.py
        └── utils.py


------------

## Folder & File Structure Requirements
Refer to Challenge page on Synapse for submission requirements
The segmentation files need to adhere to the following rules:
- Be NIfTI and use .nii.gz extension
- Dimensions should be 240 X 240 X 155 and origin at [0, -239, 0]
- Use CaPTk to verify
- Filenames should end with 5 digits case ID, followed by a dash, and 3-digit timepoint. (eg. *{ID}-{timepoint}.nii.gz

Segmentations should be contained in a zip or tarball archive and upload this to Synapse
To submit click: File Tools > Submit File to Challenge
There are 5 queues and as a team we are limited to 2 submissions per day



### Training & Validation Data
Total file: Training data
- BraTS glioma dataset = 1251
- BraTS SSA dataset = 60

All data files are labelled as follows in the new data release:
- BraTS-GLI-#####-000-#.nii.gz
- BraTS-SSA-#####-000-#.nii.gz

To run the code on your local machine, or via Compute Canada, folder structure should be set up as follows:

- source data provided: xxx/data/train/Nifty (TBC)
- source data provided: xxx/data/val/Nifty (TBC)
- Mapping file: xxx/data/BraTS2023_2017_GLI_Mapping.xlsx

data preparation outputs: 
    - data/train/subj/
        - subjxxx-stk.nii : the stacked volumes from T1n, T1c, T2w, T2f (in that order)
        - subjxxx-stk.npy : initial pre-processed stacked files == check RAS and normalise; main pre-process must include croporpad
        - subjxxx-lbl.nii : extracted seg file img data
        - subjxxx-lbl.npy : initial pre-processed seg file == MUST MATCH stk.npy transformations AT ALL TIMES

data augmentation outputs: currently nothing is saved
    - BRaTS23 GLI TRAIN DATA - some augmented to mimic poor quality data from SSA
    - BraTS23 SSA TRAIN DATA - ??? augmentations applied ??? TBD
    - BraTS23 GLI VALIDATION DATA - should be changed to BraTS fake SSA data to be used as a validation set
    - 

segmentation masks generated: TBD

## Challenge Overview
### Data
    - data sets from adult populations collected through a collaborative network of imaging centres in Africa
    - collection of pre-operative glioma data comprising of multi-parametric (mpMRI) routine clinical scans acquired as part of standard clinical care from multiple institutions and different scanners using conventional brain tumor imaging protocols
    - differences in imaging systems and variations in clinical imaging protocols = vastly heterogeneous image quality
    - ground truth annotations were approved by expert neuroradiologists and reviewed by board-certified radiologists with extensive experience in the field of neuro-oncology
    **training (70%), validation (10%), testing (20%)**
    - training data provided with associated ground truth labels, and validation data without any associated ground truth
    - image volumes of:
        - T1-weighted **(T1)**
        - post gadolinium (Gd) contrast T1-weighted **(T1Gd)**
        - T2-weighted **(T2)**
        - T2 Fluid Attenuated Inversion Recovery **(T2-FLAIR)**
    
### Standardised BraTS **pre-processing workflow** used
    - identical with pipeline from BraTS2017-2022 - publicly available
    - conversion of DICOM to files to NIfTI file format --  strips accompanying metadata from the images and removes all Protected Health Information from DICOM headers
    - cor-registration to the same anatomical template (SRI24)
     - resampling to uniform isotropic resolution (1mm^2)
     - skull stripping (uses DL approach) --  mitigates potential facial reconstruction/recognition of the patient
        
### Generation of **ground truth labels**
    - volumes segmented using STAPLE fusion of previous top-ranked BraTS algorithms (nnU-Net, DeepScan and DeepMedic)
    - segmented images refined manually by volunteer trained radiology experts of varying rank and experience
    - then two senior attending board-certified radiologists with 5 or more years experience reviewed the segmentations
    - iterative process until the approvers found the refined tumor sub-region segmentations accceptable for public release and challenge conduction
    - finally approved by experienced board-certified attending neuro-radiologists with more than 5 years experience in interpreting glioma brain MRI
        
    - **sub-regions** -- these are image-based and do not reflect strict biologic entities
        - enhancing tumor (ET)
        - non-enhancing tumor core (NETC)
        - surrounding non-enhancing flair hyperintensity (SNFH)
        
### Important information
#### Training & Val
    - training data has ground truths available
    - validation data (released 3 weeks after training data) does not have any ground truth
    - ***NB: challenge participants can supplement the data set with additional public and/or private glioma MRI data for training algorithms***
        - supplemental data set must be explicitly and thoroughly described in the methods of submitted manuscripts
        - required to report results using only the BraTS2023 glioma data and results that include the supplemental data and discuss potential result differences
    - for submission participants are required to package their developed approach in an MLCube container following provided in the Synapse platform - Cube containers are automatically generated by GaNDLF and will be used to evaluate all submissions through the MedPerf platform
    
#### Evaluation metrics
    - Dice Similarity Coefficient  
    - 95% Hausdorff distance (as opposed to standard HD in order to avoid outliers having too much weight)
    - precision (to complement sensitivity)

#### Other
    - submitted algorithms will be ranked based on the generated metric results on the test cases by computing the summation of their ranks across the average of the metrics described above as a univariate overall summary measure
    - outcome will be plotted via an augmented version of radar plot - to visualise the results
    - missing results on test cases or if an algorithm fails to produce a result metric for a specific test case the metric will be set to its worst possible value (e.g. 0 for DSC) 
    - uncertainties in rankings will be assessed using permutational analyses“Performance for the segmentation task will be assessed based on relative performance of each team on each tumor tissue class and for each segmentation measure
    - multiple submissions to the online evaluation platforms will be allowed 
    - top ranked teams in validation phase will be invited to prepare their slides for a short oral presentation of their method during the BraTS challenge at MICCAI 2023
    - “all participants will be evaluated and ranked on the same unseen testing data, which will not be made available to the participants, after uploading their containerized method in the evaluation platforms
    - final top ranked teams will be announced at MICCAI 2023 (monetary prizes)
    - participating African teams with best rank will receive Lacuna Equity & Health Prizes (limited to BraTS-Africa BrainHack teams)
