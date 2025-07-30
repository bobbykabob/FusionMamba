import torch
import numpy as np
from TaskFusion_dataset import Fusion_dataset

def check_test_classes():
    """Check what classes actually exist in the test set"""
    test_dataset = Fusion_dataset('test', dataset_name='pst900')
    
    all_classes = set()
    class_counts = {i: 0 for i in range(9)}
    
    print("Analyzing test dataset classes...")
    
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        image_vis, image_ir, target = data
        
        # Get unique classes in this sample
        unique_classes = torch.unique(target).tolist()
        all_classes.update(unique_classes)
        
        # Count pixels for each class
        for class_id in range(9):
            count = torch.sum(target == class_id).item()
            class_counts[class_id] += count
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(test_dataset)} samples...")
    
    print(f"\nAll classes found in test set: {sorted(all_classes)}")
    print(f"Number of unique classes: {len(all_classes)}")
    
    total_pixels = sum(class_counts.values())
    print(f"\nClass distribution in test set:")
    for class_id in range(9):
        if class_counts[class_id] > 0:
            percentage = (class_counts[class_id] / total_pixels) * 100
            print(f"  Class {class_id}: {class_counts[class_id]} pixels ({percentage:.2f}%)")

if __name__ == "__main__":
    check_test_classes() 