#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_msssim
import numpy as np

ssim_loss = pytorch_msssim.msssim


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)



class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,labels,generate_img,i):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        wb0 = 0.5
        wb1 = 0.5

        ssim_loss_temp1 = ssim_loss(generate_img, image_y, normalize=True)
        ssim_loss_temp2 = ssim_loss(generate_img, image_ir, normalize=True)
        ssim_value = wb0 * (1 - ssim_loss_temp1) + wb1 * (1 - ssim_loss_temp2)
        loss_in = F.mse_loss(x_in_max, generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=(100*ssim_value)+(10*loss_in)+(1*loss_grad)
        return loss_total, loss_in, ssim_value, loss_grad

#CT-MRI loss_in:10 loss_ssim:10,loss_grad:1


class SegmentationLoss(nn.Module):
    """Segmentation loss combining cross-entropy and focal loss for semantic segmentation"""
    
    def __init__(self, num_classes=9, alpha=1.0, gamma=2.0, ignore_index=-100, weight=None):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        # Calculate class weights for PST900 (inverse frequency)
        if weight is None and num_classes == 5:  # PST900 has 5 classes
            # Based on our analysis: Background=97%, Person=0.1%, Car=0.8%, Bicycle=0.1%, Motorcycle=1.8%
            class_weights = torch.tensor([1.0, 970.0, 121.25, 970.0, 53.89])  # Inverse of frequencies
            weight = class_weights
        
        # Cross-entropy loss with class weights
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        
    def focal_loss(self, pred, target):
        """Focal loss implementation for handling class imbalance"""
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: [B, num_classes, H, W] - predicted logits
            target: [B, H, W] - ground truth labels
        """
        # Cross-entropy loss
        ce_loss_value = self.ce_loss(pred_logits, target)
        
        # Focal loss for hard examples
        focal_loss_value = self.focal_loss(pred_logits, target)
        
        # Combined loss with higher focal loss weight for better handling of class imbalance
        total_loss = ce_loss_value + 0.5 * focal_loss_value
        
        return total_loss, ce_loss_value, focal_loss_value


class CombinedFusionSegmentationLoss(nn.Module):
    """Combined loss for both fusion and segmentation tasks"""
    
    def __init__(self, fusion_weight=1.0, segmentation_weight=1.0, num_seg_classes=9):
        super(CombinedFusionSegmentationLoss, self).__init__()
        self.fusion_weight = fusion_weight
        self.segmentation_weight = segmentation_weight
        
        # Initialize individual losses
        self.fusion_loss = Fusionloss()
        self.segmentation_loss = SegmentationLoss(num_classes=num_seg_classes)
        
    def forward(self, image_vis, image_ir, seg_labels, generated_outputs, i=0):
        """
        Args:
            image_vis: visible image input
            image_ir: infrared image input  
            seg_labels: segmentation ground truth labels [B, H, W]
            generated_outputs: dict with 'fusion' and 'segmentation' outputs
            i: iteration number (for fusion loss)
        
        Returns:
            dict with loss components
        """
        losses = {}
        total_loss = 0
        
        # Fusion loss (if fusion output is available)
        if 'fusion' in generated_outputs:
            fusion_output = generated_outputs['fusion']
            fusion_loss_total, loss_in, ssim_value, loss_grad = self.fusion_loss(
                image_vis, image_ir, None, fusion_output, i
            )
            losses['fusion_total'] = fusion_loss_total
            losses['fusion_mse'] = loss_in
            losses['fusion_ssim'] = ssim_value
            losses['fusion_grad'] = loss_grad
            total_loss += self.fusion_weight * fusion_loss_total
            
        # Segmentation loss (if segmentation output is available)
        if 'segmentation' in generated_outputs and seg_labels is not None:
            seg_output = generated_outputs['segmentation']
            seg_loss_total, ce_loss, focal_loss = self.segmentation_loss(seg_output, seg_labels)
            losses['seg_total'] = seg_loss_total
            losses['seg_ce'] = ce_loss
            losses['seg_focal'] = focal_loss
            total_loss += self.segmentation_weight * seg_loss_total
            
        losses['total'] = total_loss
        return losses

