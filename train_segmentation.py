#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Training script for FusionMamba with segmentation capability
This script trains the model for both image fusion and semantic segmentation tasks.
"""

from PIL import Image
import numpy as np
from glob import glob
from torch.autograd import Variable
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger

from loss import CombinedFusionSegmentationLoss, SegmentationLoss

import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='mfnet', 
                      choices=['acod', 'mfnet', 'pst900'], help='Dataset to use for segmentation training')
    parse.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parse.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parse.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parse.add_argument('--fusion_weight', type=float, default=1.0, help='Weight for fusion loss')
    parse.add_argument('--seg_weight', type=float, default=1.0, help='Weight for segmentation loss')
    parse.add_argument('--mode', type=str, default='both', 
                      choices=['fusion', 'segmentation', 'both'], 
                      help='Training mode: fusion only, segmentation only, or both')
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

def calculate_miou(pred, target, num_classes=9):
    """Calculate mean IoU for semantic segmentation"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    ious = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        intersection = np.sum(pred_binary & target_binary)
        union = np.sum(pred_binary | target_binary)
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0

def calculate_overall_miou(pred_np, target_np, num_classes=5):
    """Calculate overall mIoU across all samples - only for classes that exist in PST900"""
    ious = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        intersection = np.sum(pred_binary & target_binary)
        union = np.sum(pred_binary | target_binary)
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0

def calculate_overall_macc(pred_np, target_np, num_classes=5):
    """Calculate overall mAcc across all samples - only for classes that exist in PST900"""
    accuracies = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        if np.sum(target_binary) > 0:
            accuracy = np.sum(pred_binary & target_binary) / np.sum(target_binary)
            accuracies.append(accuracy)
    
    return np.mean(accuracies) if accuracies else 0.0

def train_segmentation(args):
    # Setup
    modelpth = 'model_last'
    Method = 'segmentation'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, exist_ok=True)
    
    # Create model path for the specific dataset
    model_file = os.path.join(modelpth, f'segmentation_model_{args.dataset}.pth')
    
    # Initialize model with appropriate number of classes
    if args.dataset == 'acod':
        num_classes = 2  # Binary segmentation for salient object detection
    else:  # mfnet, pst900
        num_classes = 9  # Multi-class semantic segmentation
    
    segmentation_model = VSSM_Fusion_Segmentation(num_seg_classes=num_classes)
    segmentation_model.cuda()
    
    # Load pretrained weights if available
    if os.path.exists(model_file):
        print(f"Loading {args.dataset} segmentation model from: {model_file}")
        try:
            segmentation_model.load_state_dict(torch.load(model_file))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training from scratch...")
    else:
        print(f"No existing model found. Training from scratch...")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=args.lr, weight_decay=1e-4)  # Added weight decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.8)], gamma=0.1)
    
    # Setup loss
    if args.mode == 'both':
        criterion = CombinedFusionSegmentationLoss(
            fusion_weight=args.fusion_weight, 
            segmentation_weight=args.seg_weight, 
            num_seg_classes=num_classes
        )
    elif args.mode == 'segmentation':
        criterion = SegmentationLoss(num_classes=num_classes)
    else:  # fusion only
        from loss import Fusionloss
        criterion = Fusionloss()
    
    criterion.cuda()
    
    # Dataset lengths
    if args.dataset == 'mfnet':
        dataset_length = 1569
    elif args.dataset == 'pst900':
        dataset_length = 597
    else:  # acod
        dataset_length = 4600
    
    # Setup dataset
    train_dataset = Fusion_dataset('train', dataset_name=args.dataset, length=dataset_length)
    print(f"Training dataset length: {len(train_dataset)}")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    # Setup logging
    setup_logger(modelpth)
    logger = logging.getLogger()
    logger.info(f"Starting segmentation training for {args.dataset}")
    logger.info(f"Mode: {args.mode}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}, Dataset length: {dataset_length}")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    segmentation_model.train()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_fusion_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_miou = 0.0
        num_batches = 0
        
        # For overall metrics calculation
        all_pred_pixels = []
        all_target_pixels = []
        
        for i, data in enumerate(train_loader, 0):
            try:
                if len(data) == 3:  # Has segmentation labels
                    image_vis, image_ir, seg_labels = data
                    seg_labels = seg_labels.cuda()
                else:  # No segmentation labels
                    image_vis, image_ir = data
                    seg_labels = None
                
                image_vis = Variable(image_vis.cuda())
                image_ir = Variable(image_ir.cuda())
                
                optimizer.zero_grad()
                
                # Forward pass
                if args.mode == 'both':
                    outputs = segmentation_model(image_vis, image_ir, 
                                               return_fusion=True, return_segmentation=True)
                    loss_dict = criterion(image_vis, image_ir, seg_labels, outputs, i)
                    loss = loss_dict['total']
                    
                    if 'fusion_total' in loss_dict:
                        epoch_fusion_loss += loss_dict['fusion_total'].item()
                    if 'seg_total' in loss_dict:
                        epoch_seg_loss += loss_dict['seg_total'].item()
                        
                elif args.mode == 'segmentation':
                    if seg_labels is not None:
                        seg_outputs = segmentation_model.forward_segmentation_only(image_vis, image_ir)
                        loss, _, _ = criterion(seg_outputs, seg_labels)
                        epoch_seg_loss += loss.item()
                    else:
                        continue  # Skip batches without segmentation labels
                        
                else:  # fusion only
                    fusion_outputs = segmentation_model.forward_fusion_only(image_vis, image_ir)
                    loss, _, _, _ = criterion(image_vis, image_ir, None, fusion_outputs, i)
                    epoch_fusion_loss += loss.item()
                
                # Calculate mIoU for segmentation
                if args.mode in ['both', 'segmentation'] and seg_labels is not None:
                    with torch.no_grad():
                        if args.mode == 'both':
                            seg_pred = torch.argmax(outputs['segmentation'], dim=1)
                        else:
                            seg_pred = torch.argmax(seg_outputs, dim=1)
                        miou = calculate_miou(seg_pred, seg_labels, num_classes)
                        epoch_miou += miou
                        
                        # Collect all predictions and targets for overall metrics
                        all_pred_pixels.extend(seg_pred.cpu().numpy().flatten())
                        all_target_pixels.extend(seg_labels.cpu().numpy().flatten())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Print progress
                if (i + 1) % 50 == 0:
                    print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        # Scheduler step
        scheduler.step()
        
        # Calculate average metrics
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_fusion_loss = epoch_fusion_loss / num_batches if epoch_fusion_loss > 0 else 0
            avg_seg_loss = epoch_seg_loss / num_batches if epoch_seg_loss > 0 else 0
            avg_miou = epoch_miou / num_batches if epoch_miou > 0 else 0
            
            # Calculate overall metrics for PST900
            overall_miou = 0.0
            overall_macc = 0.0
            if args.dataset == 'pst900' and all_pred_pixels and all_target_pixels:
                all_pred = np.array(all_pred_pixels)
                all_target = np.array(all_target_pixels)
                overall_miou = calculate_overall_miou(all_pred, all_target, num_classes=5)
                overall_macc = calculate_overall_macc(all_pred, all_target, num_classes=5)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            log_msg = (f"Epoch [{epoch+1}/{args.epochs}] - "
                      f"Total Loss: {avg_loss:.4f}")
            
            if args.mode in ['both', 'fusion']:
                log_msg += f", Fusion Loss: {avg_fusion_loss:.4f}"
            if args.mode in ['both', 'segmentation']:
                log_msg += f", Seg Loss: {avg_seg_loss:.4f}, mIoU: {avg_miou:.4f}"
                if args.dataset == 'pst900':
                    log_msg += f", Overall mIoU: {overall_miou:.4f}, Overall mAcc: {overall_macc:.4f}"
                
            log_msg += f", Time: {epoch_time:.2f}s"
            
            print(log_msg)
            logger.info(log_msg)
            
            # Save model checkpoint every epoch
            torch.save(segmentation_model.state_dict(), model_file)
            print(f"Model saved to {model_file}")
    
    # Final save
    torch.save(segmentation_model.state_dict(), model_file)
    print(f"\nTraining completed! Final model saved to {model_file}")
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    train_segmentation(args) 