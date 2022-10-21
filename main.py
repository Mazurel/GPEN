# Dataset
from DPR import DPR_512

import os
import sys
from random import randint
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# Constants

CHECKPOINTS_FOLDER = "checkpoints/"
SAMPLE_FOLDER = "samples/"
BASE_DIR = "./"
PHOTOS_DIR = "../photos/pp"
PRETRAINED_MODEL = None # "weights/GPEN-Colorization-1024.pth"

ENABLE_WANDB = True

SAVE_EVERY = 10_000
ITERATIONS = 4_000_000

SIZE = 512  # Image size
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
        while True:
            imagePath = self.images[idx // self._dpr.lightsAmount()]
            try:
                inputImage, outputImage = self._dpr.relighten(imagePath, idx % self._dpr.lightsAmount())
                break
            except AttributeError:  # Happens when loading image has failed
                raise RuntimeError(f"Error loading {imagePath} !")

        # Move channel to the front and normalize pixels
        inputImage = np.moveaxis(inputImage, 2, 0) / 255.0
        outputImage = np.moveaxis(outputImage, 2, 0) / 255.0

        # Zero center pixels
        inputImage -= 0.5
        inputImage /= 0.5

        outputImage -= 0.5
        outputImage /= 0.5

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
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="Commands", dest="command")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", type=Path, default=None, help="Pretrained model to be used")
    train_parser.add_argument("--photos", type=Path, default=None, help="Photos to be used for training.")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("image", type=Path, nargs="+", help="Image to be transformed")
    run_parser.add_argument("--model", type=Path, help="Model to be used")
    run_parser.add_argument("--out-dir", type=Path, default=None, help="Write images to directory, instead of showing them")
    run_parser.add_argument("--by-side", action="store_true", default=False)
    args = parser.parse_args()

    if args.command == "run":
        import torch
        from torch.nn import functional as F

        import cv2

        from face_model.gpen_model import FullGenerator
        from train_simple import requires_grad

        if not torch.cuda.is_available():
            raise RuntimeError("Cuda is not avaible")

        device = "cuda"

        generator = FullGenerator(
            SIZE, LATENT, N_MLP, channel_multiplier=CHANNEL_MULTIPLIER, narrow=NARROW, device=device
        ).to(device)
        requires_grad(generator, False)

        print("Loaded model: ", args.model)
        ckpt = torch.load(args.model.as_posix())
        generator.load_state_dict(ckpt['g_ema'])
        generator.eval()

        if args.out_dir is not None and not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        for image in args.image:
            initially_loaded_image = cv2.imread(image.as_posix(), cv2.IMREAD_COLOR)
            initially_loaded_image = cv2.resize(initially_loaded_image, (SIZE, SIZE))

            if args.out_dir is None:
                cv2.imshow("Input", initially_loaded_image)

            loaded_image = cv2.cvtColor(initially_loaded_image, cv2.COLOR_BGR2RGB)
            loaded_image = np.moveaxis(loaded_image, 2, 0) / 255.0
            loaded_image -= 0.5
            loaded_image /= 0.5
            loaded_image = np.array(loaded_image, dtype="float32")
            target_image = torch.from_numpy(loaded_image).to(device).unsqueeze(0)

            pred_image, _ = generator(target_image)
            pred_image = pred_image[0]
            pred_image *= 0.5
            pred_image += 0.5
            pred_image = pred_image.cpu().numpy()
            pred_image = np.moveaxis(pred_image, 0, 2) * 255
            pred_image = np.array(pred_image, "uint8")
            pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)

            if args.out_dir is None:
                cv2.imshow("Processed", pred_image)
                cv2.waitKey(0)
            elif os.path.exists(args.out_dir):
                pth = os.path.join(args.out_dir, image.name)
                if args.by_side:
                    byside_pred_image = np.zeros((SIZE, SIZE * 2, 3))
                    byside_pred_image[0:SIZE, 0:SIZE, :] = initially_loaded_image
                    byside_pred_image[0:SIZE, SIZE:2*SIZE, :] = pred_image
                    cv2.imwrite(pth, byside_pred_image)
                else:
                    cv2.imwrite(pth, pred_image)
                print(f"Written image to {pth}")

    elif args.command == "train":
        import torch
        from torch import nn, autograd, optim
        from torch.utils import data

        from face_model.gpen_model import FullGenerator, Discriminator
        from train_simple import accumulate, train, data_sampler
        from training.loss.id_loss import IDLoss
        from training import lpips

        relighted_dataset = RelightedDataset(args.photos)

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

        if args.model is not None:
            print('load model:', args.model)

            ckpt = torch.load(args.model)

            generator.load_state_dict(ckpt['g'])
            discriminator.load_state_dict(ckpt['d'])
            g_ema.load_state_dict(ckpt['g_ema'])

            g_optim.load_state_dict(ckpt['g_optim'])
            d_optim.load_state_dict(ckpt['d_optim'])

        smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
        id_loss = IDLoss(BASE_DIR, device, ckpt_dict=None)
        lpips_func = lpips.LPIPS(net='alex', version='0.1').to(device)

        dataset = relighted_dataset
        loader = data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=data_sampler(dataset, shuffle=True),
            drop_last=True,
        )

        class Args:
            save_freq = SAVE_EVERY
            iter = ITERATIONS
            sample = SAMPLE_FOLDER
            val_dir = "val"
            ckpt = CHECKPOINTS_FOLDER
            size = SIZE
            path_batch_shrink = 2
            g_reg_every = G_REG_EVERY
            path_regularize = 2
            batch = BATCH_SIZE
            d_reg_every = D_REG_EVERY
            r1 = 10
            start_iter = 0
            distributed = False
            enable_wandb = ENABLE_WANDB

        train(Args, loader, generator, discriminator, [smooth_l1_loss, id_loss], g_optim, d_optim, g_ema, lpips_func, device)
