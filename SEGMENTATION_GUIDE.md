# FusionMamba Segmentation Module Guide

This guide explains how to use the newly added segmentation capability in the FusionMamba framework.

## Overview

The FusionMamba framework has been extended with semantic segmentation capability while maintaining its original image fusion functionality. The new implementation includes:

- **SegmentationHead**: A dedicated module for multi-class semantic segmentation
- **VSSM_Fusion_Segmentation**: An extended model that can perform both fusion and segmentation
- **Advanced Loss Functions**: Including focal loss and combined fusion-segmentation loss
- **Dedicated Training Script**: For training segmentation models
- **Updated Evaluation Scripts**: With proper segmentation metrics

## Quick Start

### 1. Training a Segmentation Model

Train on MFNet dataset (9-class segmentation):
```bash
python train_segmentation.py --dataset mfnet --mode segmentation --epochs 20
```

Train on PST900 dataset:
```bash
python train_segmentation.py --dataset pst900 --mode segmentation --epochs 20
```

Train for both fusion and segmentation (multi-task):
```bash
python train_segmentation.py --dataset mfnet --mode both --epochs 20 --fusion_weight 1.0 --seg_weight 1.0
```

### 2. Evaluating Segmentation Performance

Evaluate on MFNet:
```bash
python evaluate_mfnet_segmentation.py
```

Evaluate on PST900:
```bash
python evaluate_pst900_segmentation.py
```

## Architecture Details

### SegmentationHead Module

The `SegmentationHead` is a lightweight module that converts backbone features to class logits:

```python
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.1):
        # 1x1 convolution for classification
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
```

### VSSM_Fusion_Segmentation Model

The extended model supports three modes of operation:

1. **Fusion Only**: `forward_fusion_only(x1, x2)` - Returns fused image
2. **Segmentation Only**: `forward_segmentation_only(x1, x2)` - Returns segmentation logits
3. **Both**: `forward(x1, x2, return_fusion=True, return_segmentation=True)` - Returns both outputs

```python
# Example usage
model = VSSM_Fusion_Segmentation(num_seg_classes=9)

# For segmentation only
seg_logits = model.forward_segmentation_only(ir_image, vis_image)
predictions = torch.argmax(seg_logits, dim=1)

# For both tasks
outputs = model(ir_image, vis_image, return_fusion=True, return_segmentation=True)
fused_image = outputs['fusion']
seg_logits = outputs['segmentation']
```

## Loss Functions

### SegmentationLoss
Combines cross-entropy and focal loss for handling class imbalance:
- **Cross-entropy loss**: Standard multi-class classification loss
- **Focal loss**: Focuses learning on hard examples with parameter α=1.0, γ=2.0

### CombinedFusionSegmentationLoss
Enables multi-task learning with configurable weights:
- **Fusion weight**: Controls contribution of fusion loss
- **Segmentation weight**: Controls contribution of segmentation loss

## Training Options

### Command Line Arguments

```bash
python train_segmentation.py [OPTIONS]

Options:
  --dataset       {mfnet,pst900}   Dataset to use (required)
  --epochs        INT              Number of training epochs (default: 10)
  --lr            FLOAT            Learning rate (default: 0.0002)
  --batch_size    INT              Batch size (default: 4)
  --mode          {fusion,segmentation,both}  Training mode (default: both)
  --fusion_weight FLOAT            Weight for fusion loss (default: 1.0)
  --seg_weight    FLOAT            Weight for segmentation loss (default: 1.0)
```

### Training Modes

1. **Segmentation Only** (`--mode segmentation`):
   - Trains only the segmentation head
   - Uses SegmentationLoss (cross-entropy + focal loss)
   - Best for pure segmentation tasks

2. **Fusion Only** (`--mode fusion`):
   - Trains the original fusion model
   - Uses the original Fusionloss
   - Maintains backward compatibility

3. **Multi-task** (`--mode both`):
   - Trains both fusion and segmentation simultaneously
   - Uses CombinedFusionSegmentationLoss
   - Enables feature sharing between tasks

## Evaluation Metrics

The evaluation scripts compute three key metrics:

1. **Mean Class Accuracy (mAcc)**: Average accuracy across all classes
2. **Mean Intersection over Union (mIoU)**: Average IoU across all classes  
3. **Pixel Accuracy**: Overall per-pixel classification accuracy

## File Structure

```
├── models/
│   └── vmamba_Fusion_efficross.py     # Contains new segmentation modules
├── train_segmentation.py              # Segmentation training script
├── evaluate_mfnet_segmentation.py     # MFNet evaluation
├── evaluate_pst900_segmentation.py    # PST900 evaluation  
├── loss.py                           # Updated with segmentation losses
└── model_last/
    └── segmentation/                 # Saved segmentation models
        ├── segmentation_model_mfnet.pth
        └── segmentation_model_pst900.pth
```

## Datasets

### Supported Datasets

1. **MFNet**: RGB-Thermal dataset with 9-class semantic segmentation
   - Training samples: ~1,569 image pairs
   - Classes: Background, car, person, bike, curve, car_stop, guardrail, color_cone, bump

2. **PST900**: RGB-Thermal dataset with 9-class semantic segmentation  
   - Training samples: ~597 image pairs
   - Classes: Background, fire_extinguisher, backpack, drill, survivor, vent, pipe, electric_box, rescue_randy

### Dataset Requirements

The dataset loading expects the following structure:
```
dataset/
├── rgb/          # RGB images
├── thermal/      # Thermal images  
└── labels/       # Segmentation ground truth masks
```

## Performance Tips

1. **Start with segmentation-only training** for faster convergence
2. **Use larger batch sizes** if GPU memory allows
3. **Adjust loss weights** based on task importance:
   - Higher `--seg_weight` for better segmentation
   - Higher `--fusion_weight` for better fusion
4. **Train for more epochs** (20-50) for better performance
5. **Use learning rate scheduling** for stable training

## Common Issues and Solutions

### Issue: "Model not found" during evaluation
**Solution**: Train a model first using the training script

### Issue: Poor segmentation performance  
**Solutions**:
- Train for more epochs (20-50)
- Use segmentation-only mode first
- Check dataset labels are properly loaded
- Increase segmentation loss weight

### Issue: CUDA out of memory
**Solutions**:
- Reduce batch size (`--batch_size 2` or `--batch_size 1`)
- Use gradient checkpointing in model
- Train on smaller image resolution

## Integration with Original FusionMamba

The new segmentation capability is fully backward compatible:

- Original fusion training: Use `train.py` (unchanged)
- Original fusion evaluation: Use `test.py` (unchanged)  
- New segmentation training: Use `train_segmentation.py`
- New segmentation evaluation: Use `evaluate_*_segmentation.py`

## Example Workflow

```bash
# 1. Train segmentation model on MFNet
python train_segmentation.py --dataset mfnet --mode segmentation --epochs 20

# 2. Evaluate performance
python evaluate_mfnet_segmentation.py

# 3. Train multi-task model (optional)
python train_segmentation.py --dataset mfnet --mode both --epochs 20

# 4. Compare results
python evaluate_mfnet_segmentation.py
```

## Citation

If you use the segmentation capability in your research, please cite the original FusionMamba paper and mention the segmentation extension. 