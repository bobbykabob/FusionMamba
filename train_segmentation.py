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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

try:
    from lovasz_losses import lovasz_softmax
    LOVASZ_AVAILABLE = True
except ImportError:
    LOVASZ_AVAILABLE = False

class ImprovedPST900Loss(nn.Module):
    """Data-driven loss for PST900 with LovaszSoftmax and computed class weights"""
    def __init__(self, num_classes=5, train_dataset=None):
        super().__init__()
        self.num_classes = num_classes
        # Compute class weights from training set
        if train_dataset is not None:
            print("Computing class weights from training set...")
            class_counts = np.zeros(num_classes)
            for i in range(len(train_dataset)):
                data = train_dataset[i]
                if len(data) == 3:
                    _, _, target = data
                    target_np = target.numpy().flatten()
                    for c in range(num_classes):
                        class_counts[c] += np.sum(target_np == c)
            class_freq = class_counts / class_counts.sum()
            class_weights = 1.0 / (class_freq + 1e-6)
            class_weights = class_weights / class_weights.min()  # Normalize so min=1.0
            print(f"Class weights: {class_weights}")
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights = torch.ones(num_classes)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        self.class_weights = class_weights
    def focal_loss(self, pred, target, alpha=1.0, gamma=2.0):
        ce_loss = F.cross_entropy(pred, target, ignore_index=-100, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    def dice_loss(self, pred, target):
        smooth = 1e-6
        pred_soft = F.softmax(pred, dim=1)
        dice_loss = 0
        for class_id in range(1, self.num_classes):
            pred_class = pred_soft[:, class_id, :, :]
            target_class = (target == class_id).float()
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_loss += (1 - dice)
        return dice_loss / (self.num_classes - 1)
    def forward(self, pred_logits, target):
        ce_loss = self.ce_loss(pred_logits, target)
        focal_loss = self.focal_loss(pred_logits, target)
        dice_loss = self.dice_loss(pred_logits, target)
        if LOVASZ_AVAILABLE:
            # Lovasz expects probabilities, not logits
            probas = F.softmax(pred_logits, dim=1)
            lovasz_loss = lovasz_softmax(probas, target, ignore=-100)
            total_loss = ce_loss + 0.5 * focal_loss + 0.3 * dice_loss + 0.5 * lovasz_loss
            return total_loss, ce_loss, focal_loss, dice_loss, lovasz_loss
        else:
            total_loss = ce_loss + 0.5 * focal_loss + 0.3 * dice_loss
            return total_loss, ce_loss, focal_loss, dice_loss

# Poly LR scheduler
class PolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power for base_lr in self.base_lrs]

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
    parse.add_argument('--eval_interval', type=int, default=3, help='Evaluate every N epochs')
    return parse.parse_args()

def evaluate_model(model, test_dataset, num_classes=5):
    """Evaluate model on test dataset"""
    model.eval()
    
    # Initialize metrics
    running_metrics = {'tp': [0] * num_classes, 'fp': [0] * num_classes, 'fn': [0] * num_classes}
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for i in range(min(50, len(test_dataset))):  # Evaluate on first 50 samples
            data = test_dataset[i]
            if len(data) == 3:
                image_vis, image_ir, target = data
                
                # Prepare inputs
                image_vis = image_vis.unsqueeze(0).cuda()
                image_ir = image_ir.unsqueeze(0).cuda()
                target = target.cuda()
                
                # Get prediction
                seg_logits = model.forward_segmentation_only(image_vis, image_ir)
                pred = torch.argmax(seg_logits, dim=1)
                
                # Convert to numpy
                pred_np = pred.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()
                
                # Calculate per-class metrics
                for class_id in range(num_classes):
                    pred_binary = (pred_np == class_id)
                    target_binary = (target_np == class_id)
                    
                    tp = np.sum(pred_binary & target_binary)
                    fp = np.sum(pred_binary & ~target_binary)
                    fn = np.sum(~pred_binary & target_binary)
                    
                    running_metrics['tp'][class_id] += tp
                    running_metrics['fp'][class_id] += fp
                    running_metrics['fn'][class_id] += fn
    
    # Calculate final metrics
    class_names = {0: 'Background', 1: 'Person', 2: 'Car', 3: 'Bicycle', 4: 'Motorcycle'}
    total_iou = 0
    valid_classes = 0
    
    print("\nEvaluation Results:")
    print("-" * 50)
    for class_id in range(num_classes):
        tp = running_metrics['tp'][class_id]
        fp = running_metrics['fp'][class_id]
        fn = running_metrics['fn'][class_id]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        if tp + fn > 0:  # Only count classes that exist in ground truth
            total_iou += iou
            valid_classes += 1
        
        print(f"{class_names[class_id]:<12} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")
    
    mean_iou = total_iou / valid_classes if valid_classes > 0 else 0
    print("-" * 50)
    print(f"Mean IoU: {mean_iou:.4f}")
    
    model.train()
    return mean_iou

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
        else:
            ious.append(0.0)
    
    return np.mean(ious)

def calculate_overall_miou(pred_np, target_np, num_classes=5):
    """Calculate overall mIoU across all samples"""
    ious = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        intersection = np.sum(pred_binary & target_binary)
        union = np.sum(pred_binary | target_binary)
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            ious.append(0.0)
    
    return np.mean(ious)

def calculate_overall_macc(pred_np, target_np, num_classes=5):
    """Calculate overall mean class accuracy"""
    accuracies = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        correct = np.sum(pred_binary & target_binary)
        total = np.sum(target_binary)
        
        if total > 0:
            accuracy = correct / total
            accuracies.append(accuracy)
        else:
            accuracies.append(0.0)
    
    return np.mean(accuracies)

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
    elif args.dataset == 'mfnet':
        num_classes = 9  # Multi-class semantic segmentation
    elif args.dataset == 'pst900':
        num_classes = 5  # PST900 has 5 classes (0-4)
    else:  # acod
        num_classes = 2  # Binary segmentation for salient object detection
    
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
    
    # Setup test dataset for evaluation
    test_dataset = None
    if args.dataset == 'pst900':
        test_dataset = Fusion_dataset('test', dataset_name=args.dataset)
        print(f"Test dataset length: {len(test_dataset)}")
    
    # Use smaller batch size for PST900
    batch_size = 2 if args.dataset == 'pst900' else args.batch_size
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    # Setup optimizer with better parameters for PST900
    if args.dataset == 'pst900':
        optimizer = torch.optim.AdamW(
            segmentation_model.parameters(),
            lr=5e-5,  # Lower learning rate for stability
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        scheduler = PolyLRScheduler(optimizer, max_iter=args.epochs * len(train_loader), power=0.9)
    else:
        optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.8)], gamma=0.1)
    
    # Setup loss - use improved loss for PST900
    if args.dataset == 'pst900':
        criterion = ImprovedPST900Loss(num_classes=num_classes, train_dataset=train_dataset)
        print("Using improved PST900 loss function with data-driven class weights and LovaszSoftmax (if available)")
    elif args.mode == 'both':
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
    
    # Setup logging
    setup_logger(modelpth)
    logger = logging.getLogger()
    logger.info(f"Starting segmentation training for {args.dataset}")
    logger.info(f"Mode: {args.mode}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info(f"Batch size: {batch_size}, Dataset length: {dataset_length}")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    segmentation_model.train()
    
    best_test_miou = 0.0
    
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
                    image_vis, image_ir, target_seg = data
                    target_seg = target_seg.cuda()
                else:
                    print(f"Sample {i+1} has no segmentation label, skipping...")
                    continue
                
                image_vis = image_vis.cuda()
                image_ir = image_ir.cuda()
                
                optimizer.zero_grad()
                
                # Forward pass
                if args.mode == 'both':
                    outputs = segmentation_model(image_vis, image_ir, return_fusion=True, return_segmentation=True)
                    fusion_output = outputs.get('fusion', None)
                    seg_output = outputs.get('segmentation', None)
                    
                    # Calculate combined loss
                    loss_dict = criterion(image_vis, image_ir, target_seg, outputs, i)
                    total_loss = loss_dict['total']
                    fusion_loss = loss_dict.get('fusion_total', 0.0)
                    seg_loss = loss_dict.get('seg_total', 0.0)
                    
                elif args.mode == 'segmentation':
                    seg_output = segmentation_model.forward_segmentation_only(image_vis, image_ir)
                    
                    if args.dataset == 'pst900':
                        total_loss, ce_loss, focal_loss, dice_loss = criterion(seg_output, target_seg)
                        seg_loss = total_loss
                        fusion_loss = 0.0
                    else:
                        total_loss, ce_loss, focal_loss = criterion(seg_output, target_seg)
                        seg_loss = total_loss
                        fusion_loss = 0.0
                    
                    fusion_output = None
                else:  # fusion only
                    fusion_output = segmentation_model.forward_fusion_only(image_vis, image_ir)
                    seg_output = None
                    
                    # Calculate fusion loss
                    fusion_loss_total, loss_in, ssim_value, loss_grad = criterion(image_vis, image_ir, None, fusion_output, i)
                    total_loss = fusion_loss_total
                    fusion_loss = fusion_loss_total
                    seg_loss = 0.0
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for PST900
                if args.dataset == 'pst900':
                    torch.nn.utils.clip_grad_norm_(segmentation_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate metrics for this batch
                if seg_output is not None:
                    pred_seg = torch.argmax(seg_output, dim=1)
                    batch_miou = calculate_miou(pred_seg, target_seg, num_classes)
                    
                    # Collect pixels for overall metrics
                    pred_np = pred_seg.cpu().numpy().flatten()
                    target_np = target_seg.cpu().numpy().flatten()
                    all_pred_pixels.extend(pred_np)
                    all_target_pixels.extend(target_np)
                    
                    epoch_miou += batch_miou
                
                # Update loss tracking
                epoch_loss += total_loss.item()
                epoch_fusion_loss += fusion_loss
                epoch_seg_loss += seg_loss
                num_batches += 1
                
                # Print progress every 50 batches
                if (i + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {total_loss.item():.4f}")
                    if args.dataset == 'pst900' and seg_output is not None:
                        print(f"  CE: {ce_loss.item():.4f}, Focal: {focal_loss.item():.4f}, Dice: {dice_loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch averages
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_fusion_loss = epoch_fusion_loss / num_batches if num_batches > 0 else 0.0
        avg_seg_loss = epoch_seg_loss / num_batches if num_batches > 0 else 0.0
        avg_miou = epoch_miou / num_batches if num_batches > 0 else 0.0
        
        # Calculate overall metrics
        overall_miou = 0.0
        overall_macc = 0.0
        if all_pred_pixels and all_target_pixels:
            all_pred = np.array(all_pred_pixels)
            all_target = np.array(all_target_pixels)
            overall_miou = calculate_overall_miou(all_pred, all_target, num_classes)
            overall_macc = calculate_overall_macc(all_pred, all_target, num_classes)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Total Loss: {avg_loss:.4f}, "
              f"Seg Loss: {avg_seg_loss:.4f}, "
              f"mIoU: {avg_miou:.4f}, "
              f"Overall mIoU: {overall_miou:.4f}, "
              f"Overall mAcc: {overall_macc:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Save model
        torch.save(segmentation_model.state_dict(), model_file)
        print(f"Model saved to {model_file}")
        
        # Evaluate on test set every eval_interval epochs
        if test_dataset is not None and (epoch + 1) % args.eval_interval == 0:
            test_miou = evaluate_model(segmentation_model, test_dataset, num_classes)
            print(f"Test mIoU after epoch {epoch+1}: {test_miou:.4f}")
            
            # Save best model
            if test_miou > best_test_miou:
                best_test_miou = test_miou
                best_model_path = os.path.join(modelpth, f'segmentation_model_{args.dataset}_best.pth')
                torch.save(segmentation_model.state_dict(), best_model_path)
                print(f"New best model saved! Test mIoU: {best_test_miou:.4f}")
    
    print("Training completed!")
    if test_dataset is not None:
        print(f"Best test mIoU: {best_test_miou:.4f}")

if __name__ == "__main__":
    args = parse_args()
    train_segmentation(args) 