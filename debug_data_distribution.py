import torch
import numpy as np
from TaskFusion_dataset import Fusion_dataset

def analyze_data_distribution():
    """Analyze the data distribution differences between train and test sets"""
    
    train_dataset = Fusion_dataset('train', dataset_name='pst900')
    test_dataset = Fusion_dataset('test', dataset_name='pst900')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    for dataset_name, dataset in [('Train', train_dataset), ('Test', test_dataset)]:
        print(f"\n{'='*50}")
        print(f"Analyzing {dataset_name} dataset")
        print(f"{'='*50}")
        
        # Analyze class distribution
        class_counts = {i: 0 for i in range(5)}
        total_pixels = 0
        
        # Sample analysis (first 100 samples)
        for i in range(min(100, len(dataset))):
            data = dataset[i]
            image_vis, image_ir, target = data
            
            # Count pixels for each class
            for class_id in range(5):
                count = torch.sum(target == class_id).item()
                class_counts[class_id] += count
            
            total_pixels += target.numel()
        
        print(f"Class distribution (first 100 samples):")
        for class_id in range(5):
            percentage = (class_counts[class_id] / total_pixels) * 100
            print(f"  Class {class_id}: {class_counts[class_id]} pixels ({percentage:.2f}%)")
        
        # Check if classes are present
        present_classes = [i for i in range(5) if class_counts[i] > 0]
        print(f"Present classes: {present_classes}")
        
        # Analyze image statistics
        rgb_values = []
        thermal_values = []
        
        for i in range(min(10, len(dataset))):
            data = dataset[i]
            image_vis, image_ir, target = data
            
            rgb_values.append(image_vis.mean().item())
            thermal_values.append(image_ir.mean().item())
        
        print(f"RGB mean: {np.mean(rgb_values):.4f} ± {np.std(rgb_values):.4f}")
        print(f"Thermal mean: {np.mean(thermal_values):.4f} ± {np.std(thermal_values):.4f}")

if __name__ == "__main__":
    analyze_data_distribution() 