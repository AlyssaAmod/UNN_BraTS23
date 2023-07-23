import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import monai.networks.nets as nets

def unet():
    unet = nets.UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        # channels=(16, 32, 64, 128, 256),
        # channels=(32, 64, 128, 256, 320, 320), #nnunet channels, deoth 6
        channels=(64, 96, 128, 192, 256, 384, 512), # optinet, depth 7
        strides=(2, 2, 2, 2, 2, 2,), # length should = len(channels) - 1
        kernel_size=3,
        # num_res_units=args.res_block,
        dropout=0.7
    )
    return unet


def dynUnet(args):
    return nets.DynUNet(
        dim = 3,
        in_channels = 4,
        out_channels = 4,
        filters = args.filters,
        kernels = 3,
        strides = 2,
        norm_name = (args.norm.upper(), {"affine": True}),
        act_name = ("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
        deep_supervision = True,
        deep_supr_num = 2,
        res_block = args.res_block,
        trans_bias=True)


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
""" FOR USE WITH A DIFFUSION MODEL ONLY
    timesteps = torch.randint(
        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
    ).long()

    # Get model prediction
    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
    val_loss = F.mse_loss(noise_pred.float(), noise.float())
    """