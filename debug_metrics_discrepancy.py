import torch
import numpy as np
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
from TaskFusion_dataset import Fusion_dataset

def calculate_overall_miou(pred_np, target_np, num_classes=5):
    """Calculate overall mIoU across all samples - only for classes that exist in PST900"""
    ious = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        intersection = np.sum(pred_binary & target_binary)
        union = np.sum(pred_binary | target_binary)
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0

def calculate_overall_macc(pred_np, target_np, num_classes=5):
    """Calculate overall mAcc across all samples - only for classes that exist in PST900"""
    accuracies = []
    for class_id in range(num_classes):
        pred_binary = (pred_np == class_id)
        target_binary = (target_np == class_id)
        
        if np.sum(target_binary) > 0:
            accuracy = np.sum(pred_binary & target_binary) / np.sum(target_binary)
            accuracies.append(accuracy)
    
    return np.mean(accuracies) if accuracies else 0.0

def debug_metrics_discrepancy():
    """Debug the discrepancy between training and test metrics"""
    
    # Load model
    model = VSSM_Fusion_Segmentation(num_seg_classes=9)
    model.load_state_dict(torch.load('model_last/segmentation/segmentation_model_pst900.pth'))
    model.cuda()
    model.eval()
    
    # Load both train and test datasets
    train_dataset = Fusion_dataset('train', dataset_name='pst900')
    test_dataset = Fusion_dataset('test', dataset_name='pst900')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test on both datasets
    for dataset_name, dataset in [('Train', train_dataset), ('Test', test_dataset)]:
        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset_name} dataset")
        print(f"{'='*50}")
        
        all_pred_pixels = []
        all_target_pixels = []
        class_counts = {i: 0 for i in range(5)}
        
        # Process first 50 samples for quick comparison
        for i in range(min(50, len(dataset))):
            data = dataset[i]
            image_vis, image_ir, target = data
            
            # Prepare inputs
            image_vis = image_vis.unsqueeze(0).cuda()
            image_ir = image_ir.unsqueeze(0).cuda()
            
            # Get prediction
            with torch.no_grad():
                seg_logits = model.forward_segmentation_only(image_vis, image_ir)
                pred = torch.argmax(seg_logits, dim=1)
            
            # Collect all predictions and targets
            all_pred_pixels.extend(pred.cpu().numpy().flatten())
            all_target_pixels.extend(target.cpu().numpy().flatten())
            
            # Count class distribution
            for class_id in range(5):
                class_counts[class_id] += torch.sum(target == class_id).item()
        
        # Calculate metrics
        all_pred = np.array(all_pred_pixels)
        all_target = np.array(all_target_pixels)
        
        overall_miou = calculate_overall_miou(all_pred, all_target, num_classes=5)
        overall_macc = calculate_overall_macc(all_pred, all_target, num_classes=5)
        pixel_acc = np.sum(all_pred == all_target) / len(all_pred)
        
        print(f"Overall mIoU: {overall_miou:.4f}")
        print(f"Overall mAcc: {overall_macc:.4f}")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        
        # Show class distribution
        total_pixels = sum(class_counts.values())
        print(f"\nClass distribution:")
        for class_id in range(5):
            percentage = (class_counts[class_id] / total_pixels) * 100
            print(f"  Class {class_id}: {class_counts[class_id]} pixels ({percentage:.2f}%)")
        
        # Calculate per-class IoU
        print(f"\nPer-class IoU:")
        for class_id in range(5):
            pred_binary = (all_pred == class_id)
            target_binary = (all_target == class_id)
            
            intersection = np.sum(pred_binary & target_binary)
            union = np.sum(pred_binary | target_binary)
            
            if union > 0:
                iou = intersection / union
                print(f"  Class {class_id}: {iou:.4f}")
            else:
                print(f"  Class {class_id}: 0.0000")

if __name__ == "__main__":
    debug_metrics_discrepancy() 