import monai
import nibabel
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, decollate_batch
from monai.transforms import Compose
# from monai.utils import set_determinism
from monai.utils.misc import set_determinism
from tqdm import tqdm
from monai.networks import nets
from monai.losses import DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
)
from torch.cuda.amp import GradScaler, autocast

import data_loader as dl
import os

# Set seed
args = dl.get_main_args()
set_determinism(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model architecture
model=nets.UNet(
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
    )
model.to(device)

n_channels = len(model.channels)
print(n_channels)

# Define transforms
data_transform = dl.define_transforms(n_channels)

# Load data
dataloaders = dl.load_data(args, data_transform)                            # this also saves a json splitData
train_loader, val_loader = dataloaders['train'], dataloaders['val']

# Load model checkpoint
# 

# Print out model architecture
print(model)

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


val_interval = 1
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
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# Train model --> see MONAI notebook examples
val_interval = 1 # validation can be done every n epochs
epoch_loss_list = []
val_epoch_loss_list = []
metric_values = []
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []
# use amp to accelerate training
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
        # images = batch["image"].to(device)
        # optimizer.zero_grad(set_to_none=True)
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimiser.zero_grad()

        # cast tensor to smaller memory footprint to avoid OOM
        # with autocast(enabled=True):
        with autocast():
            # # Generate random noise
            # noise = torch.randn_like(images).to(device)
            # Create timesteps
            # timesteps = torch.randint(
            #     0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            # ).long()
            # # Get model prediction
            # noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            # loss = F.mse_loss(noise_pred.float(), noise.float())

            print(inputs.shape)
            outputs = model(inputs)

            loss = criterion.forward(outputs, labels)
        
        # Calculate Loss and Update optimiser using scalar
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # loss.backward()
        # # Update optimiser
        # optimiser.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(dataloaders['val']):
                val_inputs, val_labels = batch[0].to(device), batch[1].to(device)            
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(args.data, "best_metric_model.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
#     if (epoch + 1) % val_interval == 0:
#         model.eval()
#         val_epoch_loss = 0
#         for step, batch in enumerate(dataloaders['val']):
#             inputs, labels = batch[0].to(device), batch[1].to(device)
#         # for step, batch in enumerate(dataloaders['val']):
#             # images = batch["image"].to(device)
#             # noise = torch.randn_like(images).to(device)
#             with torch.no_grad():
#                 with autocast():
#                 # with autocast(enabled=True):
#                     # timesteps = torch.randint(
#                     #     0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
#                     # ).long()

#                     # Get model prediction
#                     # noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
#                     # val_loss = F.mse_loss(noise_pred.float(), noise.float())
#                     outputs = model(inputs)
#                     val_loss = criterion.forward(outputs, labels)

#             val_epoch_loss += val_loss.item()
#             progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
#         val_epoch_loss_list.append(val_epoch_loss / (step + 1))

#         # Sampling image during training
#         # image = torch.randn((1, 1, 32, 40, 32))
#         # image = image.to(device)
#         # scheduler.set_timesteps(num_inference_steps=1000)
#         # with autocast(enabled=True):
#         #     image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler)

#         # plt.figure(figsize=(2, 2))
#         # plt.imshow(image[0, 0, :, :, 15].cpu(), vmin=0, vmax=1, cmap="gray")
#         # plt.tight_layout()
#         # plt.axis("off")
#         # plt.show()

# total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# Save checkpoint
# TODO