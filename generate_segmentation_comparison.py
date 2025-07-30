import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
from TaskFusion_dataset import Fusion_dataset

def create_segmentation_visualization(pred, target, rgb_img, thermal_img, save_path):
    """Create a visualization showing input images, ground truth, and prediction"""
    
    # Convert tensors to numpy arrays
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    
    # Convert input images to numpy arrays
    rgb_np = rgb_img.cpu().numpy().squeeze()
    thermal_np = thermal_img.cpu().numpy().squeeze()
    
    # Normalize images for display
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())
    thermal_np = (thermal_np - thermal_np.min()) / (thermal_np.max() - thermal_np.min())
    
    # Create color maps for segmentation
    colors = {
        0: [0, 0, 0],      # Background - Black
        1: [255, 0, 0],     # Person - Red
        2: [0, 255, 0],     # Car - Green
        3: [0, 0, 255],     # Bicycle - Blue
        4: [255, 255, 0],   # Motorcycle - Yellow
    }
    
    # Create colored segmentation maps
    pred_colored = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
    target_colored = np.zeros((target_np.shape[0], target_np.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        pred_colored[pred_np == class_id] = color
        target_colored[target_np == class_id] = color
    
    # Create the comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PST900 Segmentation Comparison', fontsize=16)
    
    # Input images
    axes[0, 0].imshow(rgb_np, cmap='gray')
    axes[0, 0].set_title('RGB Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(thermal_np, cmap='gray')
    axes[0, 1].set_title('Thermal Input')
    axes[0, 1].axis('off')
    
    # Ground truth
    axes[0, 2].imshow(target_colored)
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    # Prediction
    axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title('Model Prediction')
    axes[1, 0].axis('off')
    
    # Overlay prediction on RGB
    rgb_overlay = np.copy(rgb_np)
    rgb_overlay = np.stack([rgb_overlay] * 3, axis=-1)
    rgb_overlay = (rgb_overlay * 255).astype(np.uint8)
    
    # Blend prediction with RGB
    alpha = 0.6
    blended = cv2.addWeighted(rgb_overlay, 1-alpha, pred_colored, alpha, 0)
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('Prediction Overlay on RGB')
    axes[1, 1].axis('off')
    
    # Error visualization
    error_map = np.zeros_like(pred_colored)
    correct_mask = (pred_np == target_np)
    error_map[~correct_mask] = [255, 0, 255]  # Magenta for errors
    error_map[correct_mask] = [0, 255, 0]     # Green for correct
    
    axes[1, 2].imshow(error_map)
    axes[1, 2].set_title('Error Map (Green=Correct, Magenta=Error)')
    axes[1, 2].axis('off')
    
    # Add legend
    legend_elements = []
    for class_id, color in colors.items():
        if class_id == 0:
            name = 'Background'
        elif class_id == 1:
            name = 'Person'
        elif class_id == 2:
            name = 'Car'
        elif class_id == 3:
            name = 'Bicycle'
        elif class_id == 4:
            name = 'Motorcycle'
        else:
            name = f'Class {class_id}'
        
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=[c/255 for c in color], label=name))
    
    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_segmentation_comparisons():
    """Generate segmentation comparison images for PST900 dataset"""
    
    # Load model
    model = VSSM_Fusion_Segmentation(num_seg_classes=9)
    model.load_state_dict(torch.load('model_last/segmentation/segmentation_model_pst900.pth'))
    model.cuda()
    model.eval()
    
    # Load dataset
    dataset = Fusion_dataset('test', dataset_name='pst900')
    
    # Create output directory
    output_dir = 'segmentation_comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating segmentation comparisons for PST900...")
    print(f"Output directory: {output_dir}")
    
    # Generate comparisons for first 10 samples
    for i in range(min(10, len(dataset))):
        print(f"Processing sample {i+1}/10...")
        
        # Get data
        data = dataset[i]
        image_vis, image_ir, target = data
        
        # Prepare inputs
        image_vis = image_vis.unsqueeze(0).cuda()
        image_ir = image_ir.unsqueeze(0).cuda()
        target = target.cuda()
        
        # Get prediction
        with torch.no_grad():
            seg_logits = model.forward_segmentation_only(image_vis, image_ir)
            pred = torch.argmax(seg_logits, dim=1)
        
        # Create visualization
        save_path = os.path.join(output_dir, f'segmentation_comparison_{i+1:03d}.png')
        create_segmentation_visualization(pred, target, image_vis, image_ir, save_path)
        
        # Calculate metrics for this sample
        pred_np = pred.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        # Calculate IoU for each class
        ious = []
        for class_id in range(5):  # PST900 has 5 classes
            pred_binary = (pred_np == class_id)
            target_binary = (target_np == class_id)
            
            intersection = np.sum(pred_binary & target_binary)
            union = np.sum(pred_binary | target_binary)
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        avg_iou = np.mean(ious) if ious else 0.0
        
        # Calculate pixel accuracy
        pixel_acc = np.sum(pred_np == target_np) / len(pred_np)
        
        print(f"  Sample {i+1}: IoU={avg_iou:.4f}, Pixel Acc={pixel_acc:.4f}")
    
    print(f"\nSegmentation comparisons saved to {output_dir}/")
    print("Each image shows:")
    print("- RGB and Thermal inputs")
    print("- Ground truth segmentation")
    print("- Model prediction")
    print("- Prediction overlay on RGB")
    print("- Error map (green=correct, magenta=error)")

if __name__ == "__main__":
    generate_segmentation_comparisons() 