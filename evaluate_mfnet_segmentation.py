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

def calculate_miou(pred, target, num_classes=9):
    """Calculate mean IoU for semantic segmentation - average over ALL classes"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # Calculate IoU for each class
    ious = []
    
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        intersection = np.sum(pred_binary & target_binary)
        union = np.sum(pred_binary | target_binary)
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0  # No pixels of this class in either pred or target
        
        ious.append(iou)
    
    return np.mean(ious)  # Average over ALL classes, not just present ones

def calculate_macc(pred, target, num_classes=9):
    """Calculate mean class accuracy for semantic segmentation - average over ALL classes"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # Calculate accuracy for each class
    accuracies = []
    
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        if np.sum(target_binary) > 0:
            # Class recall: correctly predicted pixels / total pixels of this class in GT
            accuracy = np.sum(pred_binary & target_binary) / np.sum(target_binary)
        else:
            # Class not present in ground truth
            accuracy = 0.0
        
        accuracies.append(accuracy)
    
    return np.mean(accuracies)  # Average over ALL classes, including zeros for missing classes

def calculate_pixel_accuracy(pred, target):
    """Calculate overall pixel accuracy"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    return np.sum(pred_np == target_np) / len(target_np)

def calculate_detailed_metrics(pred, target, num_classes=9):
    """Calculate detailed per-class metrics"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    class_metrics = {}
    class_names = ['background', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
    
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        intersection = np.sum(pred_binary & target_binary)
        union = np.sum(pred_binary | target_binary)
        target_pixels = np.sum(target_binary)
        pred_pixels = np.sum(pred_binary)
        
        iou = intersection / union if union > 0 else 0.0
        recall = intersection / target_pixels if target_pixels > 0 else 0.0
        precision = intersection / pred_pixels if pred_pixels > 0 else 0.0
        
        class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        
        class_metrics[class_id] = {
            'name': class_name,
            'iou': iou,
            'recall': recall,
            'precision': precision,
            'target_pixels': target_pixels,
            'pred_pixels': pred_pixels
        }
    
    return class_metrics

def evaluate_mfnet_segmentation():
    """Evaluate MFNet segmentation using proper segmentation head"""
    print("="*60)
    print("MFNet Segmentation Evaluation with Proper Segmentation Head")
    print("="*60)
    
    # Load model with segmentation capability
    model = VSSM_Fusion_Segmentation(num_seg_classes=9)
    model.cuda()
    model.eval()
    
    # Load pretrained segmentation model if available
    segmentation_model_path = 'model_last/segmentation/segmentation_model_mfnet.pth'
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
        print(f"Train a model first using: python train_segmentation.py --dataset mfnet")
    
    # Load test dataset
    test_dataset = Fusion_dataset('test', dataset_name='mfnet')
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    
    # Evaluation metrics
    macc_scores = []
    miou_scores = []
    pixel_acc_scores = []
    
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
            pred_seg = torch.argmax(seg_logits, dim=1)
            
            # Calculate metrics
            macc = calculate_macc(pred_seg, target_seg)
            miou = calculate_miou(pred_seg, target_seg)
            pixel_acc = calculate_pixel_accuracy(pred_seg, target_seg)
            
            macc_scores.append(macc)
            miou_scores.append(miou)
            pixel_acc_scores.append(pixel_acc)
            
            # Store detailed metrics for final analysis
            if i == 0:  # Store detailed metrics for first sample as example
                detailed_metrics = calculate_detailed_metrics(pred_seg, target_seg)
            
            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")
                print(f"  Current averages - mAcc: {np.mean(macc_scores):.4f}, "
                      f"mIoU: {np.mean(miou_scores):.4f}, "
                      f"Pixel Acc: {np.mean(pixel_acc_scores):.4f}")
    
    # Calculate average metrics
    avg_macc = np.mean(macc_scores) if macc_scores else 0.0
    avg_miou = np.mean(miou_scores) if miou_scores else 0.0
    avg_pixel_acc = np.mean(pixel_acc_scores) if pixel_acc_scores else 0.0
    
    print("\n" + "="*70)
    print("MFNet Segmentation Evaluation Results (Proper Segmentation Head):")
    print("="*70)
    print(f"mAcc (Mean Class Accuracy): {avg_macc:.4f}")
    print(f"mIoU (Mean IoU): {avg_miou:.4f}")
    print(f"Pixel Accuracy: {avg_pixel_acc:.4f}")
    print(f"Number of test samples: {len(macc_scores)}")
    print("="*70)
    
    # Show detailed per-class metrics for first sample
    if 'detailed_metrics' in locals():
        print("\nPer-Class Metrics (from first test sample):")
        print("-" * 70)
        print(f"{'Class':<15} {'IoU':<8} {'Recall':<8} {'Precision':<10} {'GT Pixels':<10}")
        print("-" * 70)
        for class_id, metrics in detailed_metrics.items():
            if metrics['target_pixels'] > 0:  # Only show classes that exist in GT
                print(f"{metrics['name']:<15} {metrics['iou']:<8.4f} {metrics['recall']:<8.4f} "
                      f"{metrics['precision']:<10.4f} {metrics['target_pixels']:<10}")
        print("-" * 70)
    
    # Save results to file
    results_file = 'mfnet_segmentation_results.txt'
    with open(results_file, 'w') as f:
        f.write("MFNet Segmentation Evaluation Results (Proper Segmentation Head):\n")
        f.write("="*60 + "\n")
        f.write(f"mAcc (Mean Class Accuracy): {avg_macc:.4f}\n")
        f.write(f"mIoU (Mean IoU): {avg_miou:.4f}\n")
        f.write(f"Pixel Accuracy: {avg_pixel_acc:.4f}\n")
        f.write(f"Number of test samples: {len(macc_scores)}\n")
        f.write("="*60 + "\n")
        f.write(f"Model used: {segmentation_model_path}\n")
        f.write("Note: This evaluation uses the proper segmentation head.\n")
        f.write("For better results, train the model using: python train_segmentation.py --dataset mfnet\n")
    
    print(f"Results saved to {results_file}")
    
    if not os.path.exists(segmentation_model_path):
        print("\n" + "="*60)
        print("IMPORTANT: No trained segmentation model found!")
        print("To get meaningful results, please:")
        print("1. Train the segmentation model:")
        print("   python train_segmentation.py --dataset mfnet --mode segmentation --epochs 20")
        print("2. Then re-run this evaluation")
        print("="*60)

if __name__ == "__main__":
    evaluate_mfnet_segmentation() 