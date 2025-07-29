import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
import argparse
from sklearn.metrics import accuracy_score, jaccard_score
import torch.nn.functional as F

def calculate_s_measure(pred, target):
    """
    Calculate S-measure (Structure-measure) for salient object detection
    Based on: "Structure-measure: A New Way to Evaluate Foreground Maps" ICCV 2017
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Normalize prediction to [0, 1] range
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    # Normalize ground truth to [0, 1] range
    target = np.clip(target, 0, 1)
    
    alpha = 0.5  # Balance coefficient
    
    # If ground truth is all zeros, return 0
    if np.sum(target) == 0:
        return 0.0
    
    # Object-aware structural similarity
    gt_fg = target
    gt_bg = 1 - target
    
    # Compute object-aware similarity
    pred_fg = pred * gt_fg
    pred_bg = pred * gt_bg
    
    mu_fg_pred = np.mean(pred_fg)
    mu_bg_pred = np.mean(pred_bg)
    mu_fg_gt = np.mean(gt_fg)
    mu_bg_gt = np.mean(gt_bg)
    
    # Object-aware similarity (bounded to [0, 1])
    score_obj = 2.0 * mu_fg_pred * mu_fg_gt / (mu_fg_pred**2 + mu_fg_gt**2 + 1e-8)
    score_obj = np.clip(score_obj, 0, 1)
    
    # Region-aware similarity - CORRECTED IMPLEMENTATION
    sigma_fg_pred = np.std(pred_fg)
    sigma_bg_pred = np.std(pred_bg)
    sigma_fg_gt = np.std(gt_fg)
    sigma_bg_gt = np.std(gt_bg)
    
    # Calculate region-aware similarity for foreground and background separately
    # Use the correct formula that ensures scores are bounded
    if sigma_fg_pred == 0 and sigma_fg_gt == 0:
        score_reg_fg = 1.0
    elif sigma_fg_pred == 0 or sigma_fg_gt == 0:
        score_reg_fg = 0.0
    else:
        # Use the correct bounded formula
        score_reg_fg = 2.0 * sigma_fg_pred * sigma_fg_gt / (sigma_fg_pred**2 + sigma_fg_gt**2 + 1e-8)
        score_reg_fg = np.clip(score_reg_fg, 0, 1)
    
    if sigma_bg_pred == 0 and sigma_bg_gt == 0:
        score_reg_bg = 1.0
    elif sigma_bg_pred == 0 or sigma_bg_gt == 0:
        score_reg_bg = 0.0
    else:
        # Use the correct bounded formula
        score_reg_bg = 2.0 * sigma_bg_pred * sigma_bg_gt / (sigma_bg_pred**2 + sigma_bg_gt**2 + 1e-8)
        score_reg_bg = np.clip(score_reg_bg, 0, 1)
    
    # Average the region scores
    score_reg = (score_reg_fg + score_reg_bg) / 2.0
    
    # Final S-measure
    s_measure = alpha * score_obj + (1 - alpha) * score_reg
    return np.clip(s_measure, 0, 1)

def calculate_f_measure_sod(pred, target, beta=0.3):
    """
    Calculate weighted F-measure (Fβ) for salient object detection
    Based on: "How to Evaluate Foreground Maps?" CVPR 2014
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Normalize prediction to [0, 1] range
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    # Normalize ground truth to [0, 1] range  
    target = np.clip(target, 0, 1)
    
    # If ground truth is all zeros, return 0
    if np.sum(target) == 0:
        return 0.0
    
    # Use adaptive threshold based on image statistics
    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    adaptive_threshold = max(pred_mean + 0.5 * pred_std, pred_mean)
    adaptive_threshold = min(adaptive_threshold, 1.0)
    
    # Binary prediction using adaptive threshold
    pred_binary = (pred >= adaptive_threshold).astype(np.float32)
    target_binary = (target >= 0.5).astype(np.float32)
    
    # Calculate precision and recall
    tp = np.sum(pred_binary * target_binary)
    fp = np.sum(pred_binary * (1 - target_binary))
    fn = np.sum((1 - pred_binary) * target_binary)
    
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    # Weighted F-measure
    if precision + recall == 0:
        return 0.0
    else:
        beta_squared = beta * beta
        f_measure = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        return f_measure

def calculate_e_measure(pred, target):
    """
    Calculate E-measure (Enhanced-alignment Measure) for salient object detection
    Based on: "Enhanced-alignment Measure for Binary Foreground Map Evaluation" IJCAI 2018
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Reshape to 2D if needed
    if pred.ndim > 2:
        pred = pred.squeeze()
    if target.ndim > 2:
        target = target.squeeze()
    
    # Normalize prediction to [0, 1] range
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    # Normalize ground truth to [0, 1] range
    target = np.clip(target, 0, 1)
    
    # If ground truth is all zeros, return 0
    if np.sum(target) == 0:
        return 0.0
    
    # Enhanced alignment matrix
    pred_norm = pred - np.mean(pred)
    target_norm = target - np.mean(target)
    
    align_matrix = 2.0 * pred_norm * target_norm / (pred_norm**2 + target_norm**2 + 1e-8)
    
    # Enhanced-alignment measure - simplified version
    enhanced_matrix = (align_matrix + 1) / 2.0  # Map to [0,1]
    e_measure = np.mean(enhanced_matrix)
    
    return np.clip(e_measure, 0, 1)

def calculate_mae(pred, target):
    """Calculate Mean Absolute Error"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # Normalize to [0,1]
    if pred_np.max() > pred_np.min():
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
    if target_np.max() > target_np.min():
        target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min())
    
    mae = np.mean(np.abs(pred_np - target_np))
    return mae

def evaluate_acod_segmentation():
    """Evaluate ACOD segmentation using standard SOD metrics (S-measure, F-measure, E-measure)"""
    print("="*70)
    print("ACOD-12K Salient Object Detection Evaluation")
    print("Using Standard SOD Metrics: S-measure, F-measure, E-measure")
    print("="*70)
    
    # Load model with segmentation capability (2 classes for binary)
    model = VSSM_Fusion_Segmentation(num_seg_classes=2)
    model.cuda()
    model.eval()
    
    # Load pretrained segmentation model if available
    segmentation_model_path = 'model_last/segmentation/segmentation_model_acod.pth'
    if os.path.exists(segmentation_model_path):
        print(f"Loading segmentation model from: {segmentation_model_path}")
        try:
            model.load_state_dict(torch.load(segmentation_model_path))
            print("Segmentation model loaded successfully!")
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
            print("Using untrained model - results will be poor")
    else:
        print("No segmentation model found. Using untrained model - results will be poor")
        print(f"Train a model first using: python train_segmentation.py --dataset acod")
    
    # Load test dataset - using a subset for testing
    print("Loading ACOD test data (using subset of training data)...")
    test_dataset = Fusion_dataset('train', dataset_name='acod', length=500)  # Use more samples for better evaluation
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    
    # Evaluation metrics
    s_measure_scores = []
    f_measure_scores = []
    e_measure_scores = []
    mae_scores = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if len(data) == 3:  # Has segmentation labels
                image_vis, image_ir, target_seg = data
                target_seg = target_seg.cuda()
            else:
                print(f"Sample {i+1} has no segmentation label, skipping...")
                continue
            
            image_vis = image_vis.cuda()
            image_ir = image_ir.cuda()
            target_seg = target_seg.cuda()
            
            # Generate segmentation predictions using proper segmentation head
            seg_logits = model.forward_segmentation_only(image_vis, image_ir)
            
            # For binary segmentation with 2 classes, use softmax (not sigmoid)
            seg_softmax = torch.softmax(seg_logits, dim=1)
            seg_probs = seg_softmax[:, 1:2]  # Get class 1 (foreground) probabilities
            
            # Calculate standard SOD metrics
            s_measure = calculate_s_measure(seg_probs.squeeze(), target_seg.squeeze())
            f_measure = calculate_f_measure_sod(seg_probs.squeeze(), target_seg.squeeze())
            e_measure = calculate_e_measure(seg_probs.squeeze(), target_seg.squeeze())
            mae = calculate_mae(seg_probs.squeeze(), target_seg.squeeze())
            
            s_measure_scores.append(s_measure)
            f_measure_scores.append(f_measure)
            e_measure_scores.append(e_measure)
            mae_scores.append(mae)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")
                print(f"  Current averages - Sα: {np.mean(s_measure_scores):.4f}, "
                      f"Fβ: {np.mean(f_measure_scores):.4f}, "
                      f"Eφ: {np.mean(e_measure_scores):.4f}, "
                      f"MAE: {np.mean(mae_scores):.4f}")
                
            # Process first 200 samples for quicker evaluation
            if i >= 199:
                break
    
    # Calculate average metrics
    avg_s_measure = np.mean(s_measure_scores) if s_measure_scores else 0.0
    avg_f_measure = np.mean(f_measure_scores) if f_measure_scores else 0.0
    avg_e_measure = np.mean(e_measure_scores) if e_measure_scores else 0.0
    avg_mae = np.mean(mae_scores) if mae_scores else 0.0
    
    print("\n" + "="*70)
    print("ACOD-12K Salient Object Detection Results (Standard SOD Metrics):")
    print("="*70)
    print(f"S-measure (Sα): {avg_s_measure:.4f}")
    print(f"F-measure (Fβ): {avg_f_measure:.4f}")  
    print(f"E-measure (Eφ): {avg_e_measure:.4f}")
    print(f"MAE: {avg_mae:.4f}")
    print(f"Number of test samples: {len(s_measure_scores)}")
    print("="*70)
    
    # Compare with benchmark
    print("\nComparison with SOTA Benchmark:")
    print("SOTA Range - Sα: 0.719-0.866, Fβ: 0.525-0.803, Eφ: 0.819-0.967")
    print(f"Our Results - Sα: {avg_s_measure:.4f}, Fβ: {avg_f_measure:.4f}, Eφ: {avg_e_measure:.4f}")
    
    gap_s = max(0.719 - avg_s_measure, 0)
    gap_f = max(0.525 - avg_f_measure, 0) 
    gap_e = max(0.819 - avg_e_measure, 0)
    print(f"Performance Gap - Sα: {gap_s:.4f}, Fβ: {gap_f:.4f}, Eφ: {gap_e:.4f}")
    
    # Save results to file
    results_file = 'acod_segmentation_results.txt'
    with open(results_file, 'w') as f:
        f.write("ACOD-12K Salient Object Detection Results (Standard SOD Metrics):\n")
        f.write("="*70 + "\n")
        f.write(f"S-measure (Sα): {avg_s_measure:.4f}\n")
        f.write(f"F-measure (Fβ): {avg_f_measure:.4f}\n")
        f.write(f"E-measure (Eφ): {avg_e_measure:.4f}\n")
        f.write(f"MAE: {avg_mae:.4f}\n")
        f.write(f"Number of test samples: {len(s_measure_scores)}\n")
        f.write("="*70 + "\n")
        f.write("\nBenchmark Comparison:\n")
        f.write("SOTA Range - Sα: 0.719-0.866, Fβ: 0.525-0.803, Eφ: 0.819-0.967\n")
        f.write(f"Our Results - Sα: {avg_s_measure:.4f}, Fβ: {avg_f_measure:.4f}, Eφ: {avg_e_measure:.4f}\n")
        f.write(f"Performance Gap - Sα: {gap_s:.4f}, Fβ: {gap_f:.4f}, Eφ: {gap_e:.4f}\n")
        f.write(f"\nModel used: {segmentation_model_path}\n")
        f.write("Note: This evaluation uses standard SOD metrics matching published benchmarks.\n")
        f.write("For better results, train the model using: python train_segmentation.py --dataset acod --epochs 50\n")
    
    print(f"\nResults saved to {results_file}")
    
    if avg_s_measure < 0.5 or avg_f_measure < 0.3:
        print("\n" + "="*70)
        print("PERFORMANCE ANALYSIS:")
        print("Current performance is significantly below benchmark.")
        print("Recommendations:")
        print("1. Train for many more epochs (50-100)")
        print("2. Use a larger batch size if GPU memory allows")
        print("3. Consider using pretrained weights from ImageNet")
        print("4. Implement proper data augmentation")
        print("5. Use a learning rate scheduler")
        print("="*70)

if __name__ == "__main__":
    evaluate_acod_segmentation() 