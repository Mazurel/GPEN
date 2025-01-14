'''
This is a simplified training code of GPEN. It achieves comparable performance as in the paper.

@Created by rosinality

@Modified by yangxy (yangtao9009@gmail.com)
'''
import argparse
import math
import random
import os
import cv2
import glob
from tqdm import tqdm

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

import __init_paths
from training.data_loader.dataset_face import FaceDataset
from face_model.gpen_model import FullGenerator, Discriminator

from training.loss.id_loss import IDLoss
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from training import lpips

import wandb


def data_sampler(dataset, shuffle, distributed=False):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        try:
            for batch in loader:
                yield batch
        except Exception:
            continue


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred, loss_funcs=None, fake_img=None, real_img=None, input_img=None):
    smooth_l1_loss, id_loss = loss_funcs
    
    loss = F.softplus(-fake_pred).mean()
    loss_l1 = smooth_l1_loss(fake_img, real_img)
    loss_id, __, __ = id_loss(fake_img, real_img, input_img)
    loss += 1.0*loss_l1 + 1.0*loss_id

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def validation(model, lpips_func, args, device):
    lq_files = sorted(glob.glob(os.path.join(args.val_dir, 'lq', '*.*')))
    hq_files = sorted(glob.glob(os.path.join(args.val_dir, 'hq', '*.*')))

    assert len(lq_files) == len(hq_files)

    dist_sum = 0
    model.eval()
    for lq_f, hq_f in zip(lq_files, hq_files):
        img_lq = cv2.imread(lq_f, cv2.IMREAD_COLOR)
        img_t = torch.from_numpy(img_lq).to(device).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t/255.-0.5)/0.5
        img_t = F.interpolate(img_t, (args.size, args.size))
        img_t = torch.flip(img_t, [1])
        
        with torch.no_grad():
            img_out, __ = model(img_t)
        
            img_hq = lpips.im2tensor(lpips.load_image(hq_f)).to(device)
            img_hq = F.interpolate(img_hq, (args.size, args.size))
            dist_sum += lpips_func.forward(img_out, img_hq)
    
    return dist_sum.data/len(lq_files)


def train(args, loader, generator, discriminator, losses, g_optim, d_optim, g_ema, lpips_func, device):
    if args.enable_wandb:
        wandb.init(project="gpen")

    loader = sample_data(loader)

    pbar = range(0, args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break

        degraded_img, real_img = next(loader)

        degraded_img = degraded_img.to(device)
        fake_img, _ = generator(degraded_img)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        fake_img, _ = generator(degraded_img)
        fake_pred = discriminator(fake_img)

        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_img, _ = generator(degraded_img)
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred, losses, fake_img, real_img, degraded_img)

        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)

            fake_img, latents = generator(degraded_img, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            if args.enable_wandb:
                wandb.log({
                    "index": idx,
                    "d": d_loss_val,
                    "g": g_loss_val,
                    "r1_val": r1_val
                })

            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                )
            )

            if i % args.save_freq == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(degraded_img)
                    sample = torch.cat((degraded_img, sample, real_img), 0)

                    if not os.path.exists(args.sample):
                        os.makedirs(args.sample)

                    utils.save_image(
                        sample,
                        f'{args.sample}/{str(i).zfill(6)}.png',
                        nrow=args.batch,
                        normalize=True,
                        range=(-1, 1),
                    )

                lpips_value = validation(g_ema, lpips_func, args, device)
                print(f'{i}/{args.iter}: lpips: {lpips_value.cpu().numpy()[0][0][0][0]}')

                # De-zerocentre sample for wandb
                sample *= .5
                sample += 0.5

                if args.enable_wandb:
                    wandb.log({
                        "index": idx,
                        # "lpips": lpips_value,
                        "Sample prediction": wandb.Image(sample.cpu(), caption=f"Sample prediction for {i}th index"),
                    })

            if i and i % args.save_freq == 0:
                if not os.path.exists(args.ckpt):
                    os.makedirs(args.ckpt)
                model_path = f'{args.ckpt}/{str(i).zfill(6)}.pth'
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    model_path
                )

                if args.enable_wandb:
                    wandb.save(model_path, policy="now")
