import torch
import numpy as np
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
from TaskFusion_dataset import Fusion_dataset

def calculate_macc_detailed(pred, target, num_classes=9):
    """Calculate mAcc with detailed breakdown"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    print(f"Detailed mAcc calculation:")
    print(f"Pred unique values: {np.unique(pred_np)}")
    print(f"Target unique values: {np.unique(target_np)}")
    print()
    
    accuracies = []
    class_details = []
    
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        target_pixels = np.sum(target_binary)
        pred_pixels = np.sum(pred_binary)
        correct_pixels = np.sum(pred_binary & target_binary)
        
        if target_pixels > 0:
            accuracy = correct_pixels / target_pixels
            accuracies.append(accuracy)
            class_details.append({
                'class_id': class_id,
                'target_pixels': target_pixels,
                'pred_pixels': pred_pixels,
                'correct_pixels': correct_pixels,
                'accuracy': accuracy
            })
            print(f"Class {class_id}: target={target_pixels}, pred={pred_pixels}, correct={correct_pixels}, acc={accuracy:.4f}")
        else:
            print(f"Class {class_id}: not present in target")
    
    macc = np.mean(accuracies) if accuracies else 0.0
    print(f"\nFinal mAcc: {macc:.4f} (averaged over {len(accuracies)} classes)")
    return macc, class_details

def analyze_multiple_samples():
    """Analyze mAcc across multiple test samples"""
    model = VSSM_Fusion_Segmentation(num_seg_classes=9)
    model.cuda()
    model.eval()
    
    # Load the trained model
    model.load_state_dict(torch.load('model_last/segmentation/segmentation_model_mfnet.pth'))
    
    # Load test dataset
    test_dataset = Fusion_dataset('test', dataset_name='mfnet')
    
    print("="*60)
    print("MFNet mAcc Debug Analysis")
    print("="*60)
    
    all_maccs = []
    all_class_counts = []
    
    with torch.no_grad():
        for i in range(min(10, len(test_dataset))):  # Analyze first 10 samples
            data = test_dataset[i]
            if len(data) == 3:
                image_vis, image_ir, target_seg = data
                
                image_vis = image_vis.unsqueeze(0).cuda()
                image_ir = image_ir.unsqueeze(0).cuda()
                target_seg = target_seg.cuda()
                
                # Get prediction
                seg_logits = model.forward_segmentation_only(image_vis, image_ir)
                pred_seg = torch.argmax(seg_logits, dim=1)
                
                print(f"\n--- Sample {i+1} ---")
                macc, class_details = calculate_macc_detailed(pred_seg, target_seg)
                all_maccs.append(macc)
                
                # Count classes present in target
                target_np = target_seg.cpu().numpy().flatten()
                present_classes = len(np.unique(target_np))
                all_class_counts.append(present_classes)
                print(f"Classes present in target: {present_classes}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Average mAcc across {len(all_maccs)} samples: {np.mean(all_maccs):.4f}")
    print(f"mAcc range: {np.min(all_maccs):.4f} - {np.max(all_maccs):.4f}")
    print(f"Average classes per sample: {np.mean(all_class_counts):.2f}")
    print(f"Classes range: {np.min(all_class_counts)} - {np.max(all_class_counts)}")

if __name__ == "__main__":
    analyze_multiple_samples() 