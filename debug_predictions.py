import torch
import numpy as np
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
import matplotlib.pyplot as plt

def debug_predictions():
    """Debug what the model is actually predicting"""
    
    # Load model
    model = VSSM_Fusion_Segmentation(num_seg_classes=2)
    model.cuda()
    model.eval()
    
    # Load model weights
    model_path = 'model_last/segmentation/segmentation_model_acod.pth'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Load test data
    test_dataset = Fusion_dataset('train', dataset_name='acod', length=10)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("Debugging predictions...")
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if len(data) == 3:
                image_vis, image_ir, target_seg = data
                image_vis = image_vis.cuda()
                image_ir = image_ir.cuda()
                target_seg = target_seg.cuda()
                
                # Get raw logits
                seg_logits = model.forward_segmentation_only(image_vis, image_ir)
                print(f"\nSample {i+1}:")
                print(f"  Logits shape: {seg_logits.shape}")
                print(f"  Logits range: [{seg_logits.min().item():.4f}, {seg_logits.max().item():.4f}]")
                
                # Check individual class logits
                class0_logits = seg_logits[:, 0]  # Background
                class1_logits = seg_logits[:, 1]  # Foreground
                print(f"  Class 0 (bg) range: [{class0_logits.min().item():.4f}, {class0_logits.max().item():.4f}]")
                print(f"  Class 1 (fg) range: [{class1_logits.min().item():.4f}, {class1_logits.max().item():.4f}]")
                
                # Apply softmax (proper for multi-class)
                seg_softmax = torch.softmax(seg_logits, dim=1)
                class1_softmax = seg_softmax[:, 1]  # Foreground probability
                print(f"  Softmax class 1 range: [{class1_softmax.min().item():.4f}, {class1_softmax.max().item():.4f}]")
                print(f"  Softmax class 1 mean: {class1_softmax.mean().item():.4f}")
                
                # Apply sigmoid to class 1 (what we were doing)
                class1_sigmoid = torch.sigmoid(class1_logits)
                print(f"  Sigmoid class 1 range: [{class1_sigmoid.min().item():.4f}, {class1_sigmoid.max().item():.4f}]")
                print(f"  Sigmoid class 1 mean: {class1_sigmoid.mean().item():.4f}")
                
                # Check target
                print(f"  Target range: [{target_seg.min().item():.4f}, {target_seg.max().item():.4f}]")
                print(f"  Target mean: {target_seg.float().mean().item():.4f}")
                print(f"  Target unique values: {torch.unique(target_seg).cpu().numpy()}")
                
                # Test different thresholds for binary prediction
                for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    # Using softmax probabilities
                    pred_binary_soft = (class1_softmax > threshold).float()
                    target_binary = target_seg.float()
                    
                    # Calculate basic metrics
                    tp = torch.sum(pred_binary_soft * target_binary).item()
                    fp = torch.sum(pred_binary_soft * (1 - target_binary)).item()
                    fn = torch.sum((1 - pred_binary_soft) * target_binary).item()
                    
                    if tp + fp > 0:
                        precision = tp / (tp + fp)
                    else:
                        precision = 0.0
                        
                    if tp + fn > 0:
                        recall = tp / (tp + fn)
                    else:
                        recall = 0.0
                    
                    if precision + recall > 0:
                        f_measure = 2 * precision * recall / (precision + recall)
                    else:
                        f_measure = 0.0
                    
                    print(f"  Threshold {threshold}: P={precision:.4f}, R={recall:.4f}, F={f_measure:.4f}")
                
                if i >= 2:  # Check first 3 samples
                    break
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("1. Check if logits look reasonable")
    print("2. Compare softmax vs sigmoid outputs")
    print("3. See which threshold gives best F-measure")
    print("4. Verify target values are correct")
    print("="*60)

if __name__ == "__main__":
    debug_predictions() 