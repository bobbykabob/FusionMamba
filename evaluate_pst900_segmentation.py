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
    """Calculate mean IoU for semantic segmentation"""
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
            ious.append(iou)
        else:
            ious.append(0.0)
    
    return np.mean(ious)

def calculate_macc(pred, target, num_classes=9):
    """Calculate mean accuracy for semantic segmentation"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # Calculate accuracy for each class
    accuracies = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        if np.sum(target_binary) > 0:
            accuracy = np.sum(pred_binary & target_binary) / np.sum(target_binary)
            accuracies.append(accuracy)
        else:
            accuracies.append(0.0)
    
    return np.mean(accuracies)

def calculate_pixel_accuracy(pred, target):
    """Calculate overall pixel accuracy"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    return np.sum(pred_np == target_np) / len(target_np)

def evaluate_pst900_segmentation():
    """Evaluate PST900 segmentation using proper segmentation head"""
    print("="*60)
    print("PST900 Segmentation Evaluation with Proper Segmentation Head")
    print("="*60)
    
    # Load model with segmentation capability
    model = VSSM_Fusion_Segmentation(num_seg_classes=9)
    model.cuda()
    model.eval()
    
    # Load pretrained segmentation model if available
    segmentation_model_path = 'model_last/segmentation/segmentation_model_pst900.pth'
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
        print(f"Train a model first using: python train_segmentation.py --dataset pst900")
    
    # Load test dataset
    test_dataset = Fusion_dataset('test', dataset_name='pst900')
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
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")
                print(f"  Current averages - mAcc: {np.mean(macc_scores):.4f}, "
                      f"mIoU: {np.mean(miou_scores):.4f}, "
                      f"Pixel Acc: {np.mean(pixel_acc_scores):.4f}")
    
    # Calculate average metrics
    avg_macc = np.mean(macc_scores) if macc_scores else 0.0
    avg_miou = np.mean(miou_scores) if miou_scores else 0.0
    avg_pixel_acc = np.mean(pixel_acc_scores) if pixel_acc_scores else 0.0
    
    print("\n" + "="*60)
    print("PST900 Segmentation Evaluation Results (Proper Segmentation Head):")
    print("="*60)
    print(f"mAcc (Mean Class Accuracy): {avg_macc:.4f}")
    print(f"mIoU (Mean IoU): {avg_miou:.4f}")
    print(f"Pixel Accuracy: {avg_pixel_acc:.4f}")
    print(f"Number of test samples: {len(macc_scores)}")
    print("="*60)
    
    # Save results to file
    results_file = 'pst900_segmentation_results.txt'
    with open(results_file, 'w') as f:
        f.write("PST900 Segmentation Evaluation Results (Proper Segmentation Head):\n")
        f.write("="*60 + "\n")
        f.write(f"mAcc (Mean Class Accuracy): {avg_macc:.4f}\n")
        f.write(f"mIoU (Mean IoU): {avg_miou:.4f}\n")
        f.write(f"Pixel Accuracy: {avg_pixel_acc:.4f}\n")
        f.write(f"Number of test samples: {len(macc_scores)}\n")
        f.write("="*60 + "\n")
        f.write(f"Model used: {segmentation_model_path}\n")
        f.write("Note: This evaluation uses the proper segmentation head.\n")
        f.write("For better results, train the model using: python train_segmentation.py --dataset pst900\n")
    
    print(f"Results saved to {results_file}")
    
    if not os.path.exists(segmentation_model_path):
        print("\n" + "="*60)
        print("IMPORTANT: No trained segmentation model found!")
        print("To get meaningful results, please:")
        print("1. Train the segmentation model:")
        print("   python train_segmentation.py --dataset pst900 --mode segmentation --epochs 20")
        print("2. Then re-run this evaluation")
        print("="*60)

if __name__ == "__main__":
    evaluate_pst900_segmentation() 