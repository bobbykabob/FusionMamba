import torch
import numpy as np
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
import matplotlib.pyplot as plt
import cv2

def debug_visual_predictions():
    """Debug visual predictions to understand the noise pattern"""
    
    # Load model
    model = VSSM_Fusion_Segmentation(num_seg_classes=2)
    model.cuda()
    model.eval()
    
    # Load model weights
    model_path = 'model_last/segmentation/segmentation_model_acod.pth'
    model.load_state_dict(torch.load(model_path))
    
    # Load test data
    test_dataset = Fusion_dataset('train', dataset_name='acod', length=10)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("Debugging visual predictions...")
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if len(data) == 3:
                image_vis, image_ir, target_seg = data
                image_vis = image_vis.cuda()
                image_ir = image_ir.cuda()
                target_seg = target_seg.cuda()
                
                # Get predictions
                seg_logits = model.forward_segmentation_only(image_vis, image_ir)
                seg_softmax = torch.softmax(seg_logits, dim=1)
                seg_probs = seg_softmax[:, 1]  # Foreground probability
                
                print(f"\nSample {i+1}:")
                print(f"  Prediction shape: {seg_probs.shape}")
                print(f"  Prediction range: [{seg_probs.min().item():.6f}, {seg_probs.max().item():.6f}]")
                print(f"  Prediction mean: {seg_probs.mean().item():.6f}")
                print(f"  Prediction std: {seg_probs.std().item():.6f}")
                
                # Convert to numpy
                pred_np = seg_probs.squeeze().cpu().numpy()
                target_np = target_seg.squeeze().cpu().numpy()
                vis_np = image_vis.squeeze().cpu().numpy()
                ir_np = image_ir.squeeze().cpu().numpy()
                
                print(f"  Target range: [{target_np.min():.1f}, {target_np.max():.1f}]")
                print(f"  Target mean: {target_np.mean():.6f}")
                print(f"  Target unique: {np.unique(target_np)}")
                
                # Check for different threshold values
                for thresh in [0.05, 0.1, 0.3, 0.5]:
                    pred_binary = (pred_np > thresh).astype(np.float32)
                    tp = np.sum(pred_binary * target_np)
                    fp = np.sum(pred_binary * (1 - target_np))
                    fn = np.sum((1 - pred_binary) * target_np)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"  Thresh {thresh}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, FG%={100*pred_binary.mean():.2f}%")
                
                # Create visualization with multiple thresholds
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                # Top row: inputs and target
                axes[0,0].imshow(vis_np, cmap='gray')
                axes[0,0].set_title('RGB Image')
                axes[0,0].axis('off')
                
                axes[0,1].imshow(ir_np, cmap='hot')
                axes[0,1].set_title('Depth Image')
                axes[0,1].axis('off')
                
                axes[0,2].imshow(target_np, cmap='gray')
                axes[0,2].set_title('Ground Truth')
                axes[0,2].axis('off')
                
                axes[0,3].imshow(pred_np, cmap='viridis', vmin=0, vmax=1)
                axes[0,3].set_title(f'Prediction (Raw)\nRange: [{pred_np.min():.4f}, {pred_np.max():.4f}]')
                axes[0,3].axis('off')
                
                # Bottom row: different thresholds
                thresholds = [0.05, 0.1, 0.3, 0.5]
                for j, thresh in enumerate(thresholds):
                    pred_binary = (pred_np > thresh).astype(np.float32)
                    axes[1,j].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
                    axes[1,j].set_title(f'Thresh {thresh}\n{100*pred_binary.mean():.1f}% foreground')
                    axes[1,j].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'debug_visual_sample_{i+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved debug_visual_sample_{i+1}.png")
                
                if i >= 2:  # Debug first 3 samples
                    break
    
    print("\n" + "="*60)
    print("VISUAL DEBUGGING SUMMARY:")
    print("1. Check if predictions have reasonable ranges")
    print("2. See which threshold works best visually")
    print("3. Compare raw predictions vs thresholded outputs")
    print("4. Look for any obvious artifacts or patterns")
    print("="*60)

if __name__ == "__main__":
    debug_visual_predictions() 