#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Improved PST900 Training Script with Better Hyperparameters
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import os

from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
from TaskFusion_dataset import Fusion_dataset

def train_pst900_improved():
    """Improved training for PST900 with better hyperparameters"""
    
    # Model setup
    model = VSSM_Fusion_Segmentation(num_seg_classes=5)
    model.cuda()
    
    # Load existing model if available
    model_path = "model_last/segmentation/segmentation_model_pst900.pth"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    # Improved loss function
    from debug_pst900_performance import create_improved_loss_function
    criterion = create_improved_loss_function()
    criterion.cuda()
    
    # Better optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval
        eta_min=1e-6
    )
    
    # Dataset with augmentation
    train_dataset = Fusion_dataset('train', dataset_name='pst900', length=597)
    
    # Improved data loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,  # Smaller batch size for better gradient estimates
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Training loop
    num_epochs = 150  # More epochs
    best_loss = float('inf')
    
    print("Starting improved training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_focal_loss = 0
        epoch_dice_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, data in enumerate(progress_bar):
            if len(data) == 3:
                image_vis, image_ir, target = data
                image_vis = image_vis.cuda()
                image_ir = image_ir.cuda()
                target = target.cuda()
                
                optimizer.zero_grad()
                
                # Forward pass
                seg_logits = model.forward_segmentation_only(image_vis, image_ir)
                
                # Loss calculation
                total_loss, ce_loss, focal_loss, dice_loss = criterion(seg_logits, target)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics
                epoch_loss += total_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_focal_loss += focal_loss.item()
                epoch_dice_loss += dice_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'CE': f"{ce_loss.item():.4f}",
                    'Focal': f"{focal_loss.item():.4f}",
                    'Dice': f"{dice_loss.item():.4f}"
                })
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_ce = epoch_ce_loss / len(train_loader)
        avg_focal = epoch_focal_loss / len(train_loader)
        avg_dice = epoch_dice_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg CE: {avg_ce:.4f}")
        print(f"  Avg Focal: {avg_focal:.4f}")
        print(f"  Avg Dice: {avg_dice:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved! Loss: {best_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"model_last/segmentation/segmentation_model_pst900_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    train_pst900_improved()
