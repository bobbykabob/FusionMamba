import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
import argparse

def visualize_segmentation_results():
    """Visualize ACOD segmentation results and save as PNG"""
    print("="*60)
    print("ACOD Segmentation Visualization")
    print("="*60)
    
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
    
    # Load test dataset - using a small subset for visualization
    print("Loading ACOD test data...")
    test_dataset = Fusion_dataset('train', dataset_name='acod', length=10)  # Small subset for visualization
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    
    # Create output directory
    output_dir = 'acod_visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting visualization...")
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
            
            # Convert tensors to numpy for visualization
            vis_img = image_vis[0].cpu().numpy().transpose(1, 2, 0)  # RGB
            ir_img = image_ir[0].cpu().numpy().transpose(1, 2, 0)    # IR
            pred_mask = seg_probs[0, 0].cpu().numpy()  # Prediction mask
            target_mask = target_seg[0].cpu().numpy()  # Ground truth mask - take full 2D mask
            
            # Print shapes for debugging
            print(f"Sample {i+1} shapes - vis_img: {vis_img.shape}, ir_img: {ir_img.shape}, pred_mask: {pred_mask.shape}, target_mask: {target_mask.shape}")
            print(f"Target tensor shape: {target_seg.shape}")
            
            # Ensure masks are 2D - handle different possible shapes
            if pred_mask.ndim == 1:
                size = int(np.sqrt(pred_mask.size))
                pred_mask = pred_mask.reshape(size, size)
            
            # Handle target mask - it should already be 2D from target_seg[0]
            if target_mask.ndim == 1:
                # If it's 1D, try to make it square
                size = int(np.sqrt(target_mask.size))
                target_mask = target_mask.reshape(size, size)
            
            # Normalize images for visualization
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
            ir_img = (ir_img - ir_img.min()) / (ir_img.max() - ir_img.min())
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'ACOD Segmentation Results - Sample {i+1}', fontsize=16)
            
            # RGB Image
            axes[0, 0].imshow(vis_img)
            axes[0, 0].set_title('RGB Image')
            axes[0, 0].axis('off')
            
            # IR Image
            axes[0, 1].imshow(ir_img, cmap='gray')
            axes[0, 1].set_title('IR Image')
            axes[0, 1].axis('off')
            
            # Ground Truth Mask
            axes[1, 0].imshow(target_mask, cmap='hot', alpha=0.8)
            axes[1, 0].set_title('Ground Truth Mask')
            axes[1, 0].axis('off')
            
            # Predicted Mask
            axes[1, 1].imshow(pred_mask, cmap='hot', alpha=0.8)
            axes[1, 1].set_title('Predicted Mask')
            axes[1, 1].axis('off')
            
            # Add metrics text
            # Calculate some basic metrics for this sample
            pred_binary = (pred_mask > 0.5).astype(np.float32)
            target_binary = (target_mask > 0.5).astype(np.float32)
            
            tp = np.sum(pred_binary * target_binary)
            fp = np.sum(pred_binary * (1 - target_binary))
            fn = np.sum((1 - pred_binary) * target_binary)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            
            metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nIoU: {iou:.3f}'
            fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Save the visualization
            output_path = os.path.join(output_dir, f'acod_segmentation_sample_{i+1:03d}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization for sample {i+1} to {output_path}")
            print(f"  Sample metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, IoU: {iou:.3f}")
            
            # Only process first 5 samples for visualization
            if i >= 4:
                break
    
    print(f"\nVisualization complete! Results saved to '{output_dir}/' directory")
    print(f"Generated {min(5, len(test_dataset))} sample visualizations")
    
    # Create a summary image showing all samples
    create_summary_visualization(output_dir)

def create_summary_visualization(output_dir):
    """Create a summary image showing all generated visualizations"""
    print("Creating summary visualization...")
    
    # Get all PNG files in the output directory
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    png_files.sort()
    
    if len(png_files) == 0:
        print("No PNG files found for summary")
        return
    
    # Load all images
    images = []
    for png_file in png_files[:5]:  # Limit to 5 images
        img_path = os.path.join(output_dir, png_file)
        img = plt.imread(img_path)
        images.append(img)
    
    # Create a grid layout
    n_images = len(images)
    cols = 2
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'acod_segmentation_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary visualization saved to {summary_path}")

if __name__ == "__main__":
    visualize_segmentation_results() 