import monai
import nibabel
import numpy as np
import time

import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.transforms import Compose
# from monai.utils import set_determinism
from monai.utils.misc import set_determinism
from tqdm import tqdm
from monai.networks import nets
from torch.cuda.amp import GradScaler, autocast

import data_loader as dl

# Set seed
set_determinism(42)
args = dl.get_main_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms
data_transform = dl.define_transforms()

# Load data
# this also saves a json splitData
dataloaders = dl.load_data(args)

# Define model architecture
model=nets.UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    # channels=(4, 8, 16, 32, 64),
    channels=(32,64,128,256,320,320), #nnunet channels, deoth 6
    # channels=(64, 96, 128, 192, 256, 384, 512) # optinet, depth 7
    strides=(2, 2, 2, 2),
    # kernel_size=,
    # num_res_units=,
    # dropout=0.0,
    )

model.to(device)

# Print out model architecture
print(model)

# Define optimiser
optimiser = args.optimiser
# Define loss function
criterion = args.criterion

# Train model
val_interval = 1 # validation can be done every n epochs
epoch_loss_list = []
val_epoch_loss_list = []

# scaler = GradScaler()
total_start = time.time()
for epoch in range(args.epochs):
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

        # with autocast(enabled=True):
        with autocast():

            # Generate random noise
            # noise = torch.randn_like(images).to(device)
            # Create timesteps
            # timesteps = torch.randint(
            #     0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            # ).long()
            # Get model prediction
            # noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            # loss = F.mse_loss(noise_pred.float(), noise.float())
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # scaler.scale(loss).backward()
        # scaler.step(optimiser)
        # scaler.update()
        loss.backward()
        # Update optimiser
        optimiser.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(dataloaders['val']):
            images = batch["image"].to(device)
            noise = torch.randn_like(images).to(device)
            with torch.no_grad():
                with autocast():
                # with autocast(enabled=True):
                    # timesteps = torch.randint(
                    #     0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    # ).long()

                    # Get model prediction
                    # noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    # val_loss = F.mse_loss(noise_pred.float(), noise.float())
                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

        # Sampling image during training
        # image = torch.randn((1, 1, 32, 40, 32))
        # image = image.to(device)
        # scheduler.set_timesteps(num_inference_steps=1000)
        # with autocast(enabled=True):
        #     image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler)

        # plt.figure(figsize=(2, 2))
        # plt.imshow(image[0, 0, :, :, 15].cpu(), vmin=0, vmax=1, cmap="gray")
        # plt.tight_layout()
        # plt.axis("off")
        # plt.show()

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")