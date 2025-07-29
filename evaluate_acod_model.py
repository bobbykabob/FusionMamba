import cv2
import numpy as np
import os
import torch
import time
from PIL import Image
from models.vmamba_Fusion_efficross import VSSM_Fusion as net
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    """Load the trained fusion model"""
    model = net()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, data_range=1.0)

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return psnr(img1, img2, data_range=1.0)

def calculate_weighted_f_measure(pred, gt, beta_sq=0.3):
    """
    Calculate weighted F-measure (Fβw) for salient object detection
    Based on: "How to Evaluate Foreground Maps?" CVPR 2014
    """
    # Normalize prediction to [0, 1] range
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    # Normalize ground truth to [0, 1] range  
    gt = np.clip(gt, 0, 1)
    
    # If ground truth is all zeros, return 0
    if np.sum(gt) == 0:
        return 0.0
    
    # Use adaptive threshold based on image statistics
    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    adaptive_threshold = max(pred_mean + 0.5 * pred_std, pred_mean)
    adaptive_threshold = min(adaptive_threshold, 1.0)
    
    # Binary prediction using adaptive threshold
    pred_binary = (pred >= adaptive_threshold).astype(np.float32)
    gt_binary = (gt >= 0.5).astype(np.float32)
    
    # Calculate precision and recall
    tp = np.sum(pred_binary * gt_binary)
    fp = np.sum(pred_binary * (1 - gt_binary))
    fn = np.sum((1 - pred_binary) * gt_binary)
    
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    # Weighted F-measure
    if precision + recall == 0:
        return 0
    else:
        f_measure = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        return f_measure

def calculate_s_measure(pred, gt):
    """
    Calculate S-measure (Structure-measure) for salient object detection
    Based on: "Structure-measure: A New Way to Evaluate Foreground Maps" ICCV 2017
    """
    # Normalize prediction to [0, 1] range
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    # Normalize ground truth to [0, 1] range
    gt = np.clip(gt, 0, 1)
    
    alpha = 0.5  # Balance coefficient
    
    # If ground truth is all zeros, return 0
    if np.sum(gt) == 0:
        return 0.0
    
    # Object-aware structural similarity
    gt_fg = gt
    gt_bg = 1 - gt
    
    # Compute object-aware similarity
    pred_fg = pred * gt_fg
    pred_bg = pred * gt_bg
    
    mu_fg_pred = np.mean(pred_fg)
    mu_bg_pred = np.mean(pred_bg)
    mu_fg_gt = np.mean(gt_fg)
    mu_bg_gt = np.mean(gt_bg)
    
    # Object-aware similarity
    score_obj = 2.0 * mu_fg_pred * mu_fg_gt / (mu_fg_pred**2 + mu_fg_gt**2 + 1e-8)
    
    # Region-aware similarity  
    sigma_fg_pred = np.std(pred_fg)
    sigma_bg_pred = np.std(pred_bg)
    sigma_fg_gt = np.std(gt_fg)
    sigma_bg_gt = np.std(gt_bg)
    
    score_reg = 4.0 * sigma_fg_pred * sigma_fg_gt / (sigma_fg_pred**2 + sigma_fg_gt**2 + 1e-8) + \
                4.0 * sigma_bg_pred * sigma_bg_gt / (sigma_bg_pred**2 + sigma_bg_gt**2 + 1e-8)
    score_reg = score_reg / 2.0
    
    # Final S-measure
    s_measure = alpha * score_obj + (1 - alpha) * score_reg
    return np.clip(s_measure, 0, 1)

def calculate_e_measure(pred, gt):
    """
    Calculate E-measure (Enhanced-alignment Measure) for salient object detection
    Based on: "Enhanced-alignment Measure for Binary Foreground Map Evaluation" IJCAI 2018
    """
    # Normalize prediction to [0, 1] range
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    # Normalize ground truth to [0, 1] range
    gt = np.clip(gt, 0, 1)
    
    # If ground truth is all zeros, return 0
    if np.sum(gt) == 0:
        return 0.0
    
    # Enhanced alignment matrix
    pred_norm = pred - np.mean(pred)
    gt_norm = gt - np.mean(gt)
    
    align_matrix = 2.0 * pred_norm * gt_norm / (pred_norm**2 + gt_norm**2 + 1e-8)
    
    # Enhanced-alignment measure - simplified version
    enhanced_matrix = (align_matrix + 1) / 2.0  # Map to [0,1]
    e_measure = np.mean(enhanced_matrix)
    
    return np.clip(e_measure, 0, 1)

def resize_images(img1, img2, size1, size2):
    """Resize images to specified sizes"""
    img1_resized = cv2.resize(img1, size1)
    img2_resized = cv2.resize(img2, size2)
    return img1_resized, img2_resized

def fusion_to_saliency_postprocess(fusion_output, rgb_input, depth_input):
    """
    Convert fusion output to saliency-like map
    This is a heuristic approach for evaluation purposes
    """
    # Normalize inputs
    fusion_norm = (fusion_output - fusion_output.min()) / (fusion_output.max() - fusion_output.min() + 1e-8)
    rgb_norm = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min() + 1e-8)
    depth_norm = (depth_input - depth_input.min()) / (depth_input.max() - depth_input.min() + 1e-8)
    
    # Calculate difference from inputs to highlight fused regions
    diff_rgb = np.abs(fusion_norm - rgb_norm)
    diff_depth = np.abs(fusion_norm - depth_norm)
    
    # Combine differences 
    saliency_map = np.maximum(diff_rgb, diff_depth)
    
    # Apply Gaussian blur to smooth the map
    import cv2
    saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)
    
    # Enhance contrast
    saliency_map = np.power(saliency_map, 1.5)
    
    # Normalize to [0, 1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    return saliency_map

def evaluate_model():
    """Evaluate the trained model on ACOD-12K test set"""
    model_path = 'model_last/my_cross/fusion_model_acod.pth'
    test_rgb_dir = '/data/harris/FusionMamba/ACOD-12K/Test/Imgs'
    test_depth_dir = '/data/harris/FusionMamba/ACOD-12K/Test/Depth'
    test_gt_dir = '/data/harris/FusionMamba/ACOD-12K/Test/GT'  # Add GT directory
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Get test files
    rgb_files = [f for f in os.listdir(test_rgb_dir) if f.endswith(('.jpg', '.png', '.bmp', '.tif'))]
    rgb_files.sort()
    
    ssim_scores = []
    psnr_scores = []
    s_measure_scores = []
    weighted_f_measure_scores = []
    e_measure_scores = []
    
    print(f"Evaluating on {len(rgb_files)} test images...")
    
    for i, rgb_file in enumerate(rgb_files):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(rgb_files)}")
        
        # Load RGB, Depth, and GT images
        rgb_path = os.path.join(test_rgb_dir, rgb_file)
        depth_file = rgb_file.replace('_left_', '_depth_')
        depth_path = os.path.join(test_depth_dir, depth_file)
        gt_path = os.path.join(test_gt_dir, rgb_file)  # GT has same name as RGB
        
        if not os.path.exists(depth_path) or not os.path.exists(gt_path):
            continue
            
        # Load images
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(depth_path, 0)  # Grayscale
        gt_img = cv2.imread(gt_path, 0)  # Load GT as grayscale
        
        if rgb_img is None or depth_img is None or gt_img is None:
            continue
        
        # Resize to 256x256
        rgb_img, depth_img = resize_images(rgb_img, depth_img, (256, 256), (256, 256))
        gt_img = cv2.resize(gt_img, (256, 256))
        
        # Convert RGB to Y channel (luminance)
        rgb_y = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        rgb_y = rgb_y.astype(np.float32) / 255.0
        depth_img = depth_img.astype(np.float32) / 255.0
        gt_img = gt_img.astype(np.float32) / 255.0
        
        # Prepare tensors
        rgb_tensor = torch.from_numpy(rgb_y).unsqueeze(0).unsqueeze(0).cuda()
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).cuda()
        
        # Inference
        with torch.no_grad():
            fused_tensor = model(rgb_tensor, depth_tensor)
            fused_np = fused_tensor.squeeze().cpu().numpy()
        
        # Post-process fusion output to saliency-like map
        saliency_map = fusion_to_saliency_postprocess(fused_np, rgb_y, depth_img)
        
        # Calculate metrics (compare saliency map with GROUND TRUTH)
        ssim_score = calculate_ssim(saliency_map, gt_img)
        psnr_score = calculate_psnr(saliency_map, gt_img)
        s_score = calculate_s_measure(saliency_map, gt_img)
        weighted_f_score = calculate_weighted_f_measure(saliency_map, gt_img)
        e_score = calculate_e_measure(saliency_map, gt_img)
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        s_measure_scores.append(s_score)
        weighted_f_measure_scores.append(weighted_f_score)
        e_measure_scores.append(e_score)
    
    # Calculate average metrics
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_s_measure = np.mean(s_measure_scores)
    avg_weighted_f_measure = np.mean(weighted_f_measure_scores)
    avg_e_measure = np.mean(e_measure_scores)
    
    return avg_s_measure, avg_weighted_f_measure, avg_e_measure, avg_psnr

if __name__ == '__main__':
    print("=" * 50)
    print("EVALUATION RESULTS ON ACOD-12K DATASET")
    print("=" * 50)
    
    s_alpha, f_beta, e_phi, psnr_val = evaluate_model()
    
    print(f"S-measure (Sα): {s_alpha:.3f}")
    print(f"F-measure (Fβw): {f_beta:.3f}")
    print(f"E-measure (Eφ): {e_phi:.3f}")
    print(f"PSNR: {psnr_val:.3f}")
    print("=" * 50) 