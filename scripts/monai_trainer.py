# Import general libraries
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import logging

# Import github scripts
import data_loader as dl

# import torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Import MONAI libraries                <--- CLEAN UP THESE IMPORTS ONCE WE KNOW WHAT libraries are used
import monai
from monai.config import print_config
from monai.data import ArrayDataset, decollate_batch, DataLoader
from monai.handlers import (
    CheckpointLoader,
    IgniteMetric,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.metrics import DiceMetric, LossMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceFocalLoss
from monai.networks import nets as monNets
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose
)
from monai.inferers import sliding_window_inference
from monai.utils import first
from monai.utils.misc import set_determinism

# Other imports (unsure)
import ignite
import nibabel

"""General Setup: 
    logging,
    utils.args 
    seed,
    cuda, 
    root dir"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
args = dl.get_main_args()
set_determinism(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = args.data
"""
Potentially useful functions for model tracking and checkpoint loading
"""
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


"""Define model architecture:
        Done before data loader so that transforms has n_channels for EnsureShapeMultiple
"""
model=UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    # channels=(32, 64, 128, 256, 320, 320), #nnunet channels, deoth 6
    # channels=(64, 96, 128, 192, 256, 384, 512) # optinet, depth 7
    strides=(2, 2, 2, 2), # length should = len(channels) - 1
    # kernel_size=,
    # num_res_units=,
    # dropout=0.0,
    ).to(device)
n_channels = len(model.channels)
print(n_channels)

"""Setup transforms, dataset"""
# Define transforms
data_transform = dl.define_transforms(n_channels)
# Load data
dataloaders = dl.load_data(args, data_transform)                            # this also saves a json splitData
train_loader, val_loader = dataloaders['train'], dataloaders['val']

"""Create Model Params:
    optimiser
    loss fn
    lr
"""
# Print out model architecture
print(model)

# Load model checkpoint
# 

# Define optimiser
if args.optimiser == "adam":
    optimiser = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    print("Adam optimizer set")
elif args.optimiser == "sgd":
    optimiser = torch.optim.SGD(params=model.parameters())
    print("SGD optimizer set")
elif args.optimiser == "novo":
    optimiser = monai.optimizers.Novograd(params=model.parameters(), lr=args.learning_rate)
else:
    print("Error, no optimiser provided")

# Define loss function
if args.criterion == "ce":
    criterion = nn.CrossEntropyLoss()
    print("Cross Entropy Loss set")
elif args.criterion == "dice":
    criterion = DiceFocalLoss(squared_pred=True, to_onehot_y=False, sigmoid=True)
    print("Focal-Dice Loss set")
else:
    print("Error, no loss fn provided")

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

"""
Setup validation stuff
    metrics
    post trans ???????
    define inference
"""
VAL_AMP = True
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

"""
Define training loop
    initialise empty lists for val
    Add GradScalar which uses automatic mixed precision to accelerate training
    forward and backward passes
    validate training epoch

"""
# Train model --> see MONAI notebook examples
val_interval = 1
epoch_loss_list = []
val_epoch_loss_list = []

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]

metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

scaler = GradScaler()

total_start = time.time()

for epoch in range(args.epochs):
    epoch_start = time.time()
    # print("-" * 10)
    # print(f"epoch {epoch + 1}/{args.epochs}")
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(dataloaders['train']), total=len(dataloaders['train']), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")

    # for step, batch in progress_bar:
    for step, batch_data in progress_bar:
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimiser.zero_grad(set_to_none=True)

        # cast tensor to smaller memory footprint to avoid OOM
        with autocast(enabled=True):
            """ FOR USE WITH A DIFFUSION MODEL ONLY
            # Generate random noise
            noise = torch.randn_like(images).to(device)
            Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()
            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())
             """

            print(inputs.shape)
            outputs = model(inputs)

            loss = criterion.forward(outputs, labels)
        
        # Calculate Loss and Update optimiser using scalar
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(dataloaders['val']):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    """ FOR USE WITH A DIFFUSION MODEL ONLY
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                    # Get model prediction
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())
                    """
                    outputs = model(inputs)
                    val_loss = criterion.forward(outputs, labels)

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

total_time = time.time() - total_start


# Save checkpoint
# TODO