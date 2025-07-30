#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from glob import glob
from torch.autograd import Variable
from models.vmamba_Fusion_efficross import VSSM_Fusion
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger


from loss import Fusionloss

import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_fusion(num=0, logger=None, dataset_name='acod', epochs=1):
    lr_start = 0.0002
    modelpth = 'model_last'
    Method = 'my_cross'
    modelpth = os.path.join(modelpth, Method)
    
    # Create dataset-specific model paths
    acod_model_path = os.path.join(modelpth, 'fusion_model_acod.pth')
    mfnet_model_path = os.path.join(modelpth, 'fusion_model_mfnet.pth')
    pst900_model_path = os.path.join(modelpth, 'fusion_model_pst900.pth')
    
    fusionmodel = eval('VSSM_Fusion')()
    fusionmodel.cuda()
    
    # Load dataset-specific model if it exists
    if dataset_name == 'acod':
        model_file = acod_model_path
        dataset_length = 4600
    elif dataset_name == 'mfnet':
        model_file = mfnet_model_path
        dataset_length = 1569  # Actual number of valid MFNet image pairs
    else:  # pst900
        model_file = pst900_model_path
        dataset_length = 597  # Actual number of valid PST900 image pairs
    
    if os.path.exists(model_file):
        print(f"Loading {dataset_name} model from: {model_file}")
        fusionmodel.load_state_dict(torch.load(model_file))
        logger.info(f"Loaded {dataset_name} model from: {model_file}")
    else:
        print(f"No {dataset_name} model found, starting from scratch")
        logger.info(f"No {dataset_name} model found, starting from scratch")
    
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('train', length=dataset_length, dataset_name=dataset_name)
    print(f"Training on {dataset_name} dataset, length: {train_dataset.length}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=6,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss()

    epoch = epochs
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        lr_start = 0.0001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, data in enumerate(train_loader):
            if len(data) == 3:
                image_vis, image_ir, labels = data
            else:
                image_vis, image_ir = data
                labels = None
            try:
                fusionmodel.train()
                image_vis = Variable(image_vis).cuda()
                # image_vis_ycrcb = image_vis[:,0:1:,:,:]
                image_ir = Variable(image_ir).cuda()
                fusion_image = fusionmodel(image_vis, image_ir)

            except TypeError as e:
                print(f"Caught TypeError: {e}")


            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            optimizer.zero_grad()


            # fusion loss
            loss_fusion,  loss_in, ssim_loss, loss_grad= criteria_fusion(
                image_vis=image_vis, image_ir=image_ir, generate_img=
                fusion_image, i=num, labels=None
            )



            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'ssim_loss: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=ssim_loss.item(),
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
    # Save dataset-specific model
    if dataset_name == 'acod':
        save_path = acod_model_path
    elif dataset_name == 'mfnet':
        save_path = mfnet_model_path
    else:  # pst900
        save_path = pst900_model_path
    
    torch.save(fusionmodel.state_dict(), save_path)
    logger.info(f"{dataset_name.upper()} Model Save to: {save_path}")
    logger.info('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='VSSM_Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    parser.add_argument('--dataset', '-d', type=str, default='all', choices=['acod', 'mfnet', 'pst900', 'all'])
    parser.add_argument('--epochs', '-e', type=int, default=1)
    args = parser.parse_args()
    
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    
    if args.dataset == 'all':
        # Train all three datasets sequentially
        print("Training ACOD-12K dataset...")
        train_fusion(0, logger, 'acod', args.epochs)
        print("|1 ACOD-12K Train Fusion Model Successfully~!")
        
        print("Training MFNet dataset...")
        train_fusion(1, logger, 'mfnet', args.epochs)
        print("|2 MFNet Train Fusion Model Successfully~!")
        
        print("Training PST900 dataset...")
        train_fusion(2, logger, 'pst900', args.epochs)
        print("|3 PST900 Train Fusion Model Successfully~!")
    else:
        # Train single dataset
        train_fusion(0, logger, args.dataset, args.epochs)
        print(f"|{args.dataset.upper()} Train Fusion Model Successfully~!")
    
    print("training Done!")
