# Dataset
from DPR import DPR_512

import os
import sys
from random import randint

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, autograd, optim
from torch.utils import data

# GPEN:
from face_model.gpen_model import FullGenerator, Discriminator
from train_simple import accumulate, train, data_sampler
from training.loss.id_loss import IDLoss
from training import lpips

# Constants

CHECKPOINTS_FOLDER = "checkpoints/"
SAMPLE_FOLDER = "samples/"
BASE_DIR = "./"
PRETRAINED_MODEL = None # "weights/GPEN-Colorization-1024.pth"

SIZE = 64
LATENT = 32
N_MLP = 2
CHANNEL_MULTIPLIER = 2
NARROW = 1.0

LR = 0.002
BATCH_SIZE = 2
G_REG_EVERY = 4
D_REG_EVERY = 16

class RelightedDataset:
    '''
    This dataset contains all images provided by the path and returns them
    with associated relightened version.
    '''

    def __init__(self, path="./photos/"):
        # Load images path
        images = os.listdir(path)
        images = filter(RelightedDataset._checkValidExtension, images)
        images = map(lambda f: os.path.abspath(os.path.join(path, f)), images)
        self.images = sorted(images)

        # Initialize DPR
        self._dpr = DPR_512((SIZE, SIZE))

    def __len__(self):
        return len(self.images) * self._dpr.lightsAmount()

    def __getitem__(self, idx):
        # NOTE: Each image has variation of all possible lights
        imagePath = self.images[idx // self._dpr.lightsAmount()]
        inputImage, outputImage = self._dpr.relighten(imagePath, idx % self._dpr.lightsAmount())

        # Reorder axis for pytorch and rescale values
        inputImage = np.moveaxis(inputImage, 2, 0) / 255.0
        outputImage = np.moveaxis(outputImage, 2, 0) / 255.0

        inputImage = np.array(inputImage, dtype=np.float32)
        outputImage = np.array(outputImage, dtype=np.float32)
        #      In:          Out:
        return outputImage, inputImage

    @staticmethod
    def _checkValidExtension(file):
        filename = os.path.basename(file)
        extension = filename.split(".")[-1]
        return extension in ["jpg", "png"]


def input_output_grid(data):
    LEN = 8

    width = LEN * SIZE
    height = SIZE * 2

    outMat = np.zeros((height, width, 3), dtype=np.float32)

    for i, j in enumerate([randint(0, len(data)) for _ in range(LEN)]):
        inp, out = data[j]
        outMat[0:SIZE, i*SIZE:(i+1)*SIZE, :] = out[:, :, :] / 255
        outMat[SIZE:2*SIZE, i*SIZE:(i+1)*SIZE, :] = inp[:, :, :] / 255

    return outMat


if __name__ == "__main__":
    relighted_dataset = RelightedDataset("../photos/pp/")

    os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)
    os.makedirs(SAMPLE_FOLDER, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is not avaible")

    device = "cuda"

    generator = FullGenerator(
        SIZE, LATENT, N_MLP, channel_multiplier=CHANNEL_MULTIPLIER, narrow=NARROW, device=device
    ).to(device)
    discriminator = Discriminator(
        SIZE, channel_multiplier=CHANNEL_MULTIPLIER, narrow=NARROW, device=device
    ).to(device)
    g_ema = FullGenerator(
        SIZE, LATENT, N_MLP, channel_multiplier=CHANNEL_MULTIPLIER, narrow=NARROW, device=device
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = G_REG_EVERY / (G_REG_EVERY + 1)
    d_reg_ratio = D_REG_EVERY / (D_REG_EVERY + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=LR * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=LR * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if PRETRAINED_MODEL is not None:
        print('load model:', PRETRAINED_MODEL)

        ckpt = torch.load(PRETRAINED_MODEL)

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])


    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
    id_loss = IDLoss(BASE_DIR, device, ckpt_dict=None)
    lpips_func = lpips.LPIPS(net='alex',version='0.1').to(device)

    dataset = relighted_dataset
    loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=data_sampler(dataset, shuffle=True),
        drop_last=True,
    )

    class Args:
        save_freq = 10000
        iter = 4_000_000
        sample = "sample"
        path_batch_shrink = 2
        g_reg_every = G_REG_EVERY
        path_regularize = 2
        batch = BATCH_SIZE
        d_reg_every = D_REG_EVERY
        r1 = 10
        start_iter = 0
        distributed = False

    train(Args, loader, generator, discriminator, [smooth_l1_loss, id_loss], g_optim, d_optim, g_ema, lpips_func, device)
