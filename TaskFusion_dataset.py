import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
from numpy import asarray

def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = asarray(arr)
    im = Image.fromarray(numpydata, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, length=0, dataset_name='acod'):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.filepath_labels = []
        self.filenames_labels = []
        self.length = length  # This place can be set up as much as you want to train
        self.dataset_name = dataset_name
        
        if split == 'train':
            if dataset_name == 'acod':
                self._load_acod_data()
            elif dataset_name == 'mfnet':
                self._load_mfnet_data()
            elif dataset_name == 'pst900':
                self._load_pst900_data()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            self.split = split
        elif split == 'test':
            if dataset_name == 'pst900':
                self._load_pst900_test_data()
            elif dataset_name == 'mfnet':
                self._load_mfnet_test_data()
            else:
                data_dir_vis = vi_path
                data_dir_ir = ir_path
                self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
                self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split

    def _load_acod_data(self):
        """Load ACOD-12K dataset"""
        data_dir_base = "/data/harris/FusionMamba/ACOD-12K/Train"
        data_dir_vis = os.path.join(data_dir_base, "Imgs")  # RGB images
        data_dir_ir = os.path.join(data_dir_base, "Depth")  # Depth images
        data_dir_gt = os.path.join(data_dir_base, "GT")  # Ground truth
        
        # Get all image files from Imgs directory
        vis_files = [f for f in os.listdir(data_dir_vis) if f.endswith(('.jpg', '.png', '.bmp', '.tif'))]
        vis_files.sort()
        
        for file in vis_files:
            filepath_vis_ = os.path.join(data_dir_vis, file)
            # Convert filename from object_left_XXXXX.png to object_depth_XXXXX.png
            depth_filename = file.replace('_left_', '_depth_')
            filepath_ir_ = os.path.join(data_dir_ir, depth_filename)
            filepath_gt_ = os.path.join(data_dir_gt, file)  # GT has same name as RGB
            
            # Check if corresponding depth and GT files exist
            if os.path.exists(filepath_ir_) and os.path.exists(filepath_gt_):
                self.filepath_vis.append(filepath_vis_)
                self.filenames_vis.append(file)
                self.filepath_ir.append(filepath_ir_)
                self.filenames_ir.append(file)
                self.filepath_labels.append(filepath_gt_)
                self.filenames_labels.append(file)

    def _load_mfnet_data(self):
        """Load MFNet dataset"""
        data_dir_base = "/data/harris/FusionMamba/MFNet/ir_seg_dataset"
        data_dir_vis = os.path.join(data_dir_base, "images")  # RGB images
        data_dir_ir = os.path.join(data_dir_base, "visual")   # Thermal images
        data_dir_labels = os.path.join(data_dir_base, "labels") # Segmentation labels
        
        # Get all image files from images directory (RGB images are .png)
        vis_files = [f for f in os.listdir(data_dir_vis) if f.endswith('.png')]
        vis_files.sort()
        
        print(f"Found {len(vis_files)} RGB images in MFNet dataset")
        
        for file in vis_files:
            filepath_vis_ = os.path.join(data_dir_vis, file)
            # Convert .png to .jpg for thermal images
            thermal_file = file.replace('.png', '.jpg')
            filepath_ir_ = os.path.join(data_dir_ir, thermal_file)
            
            # Check if corresponding thermal file exists
            if os.path.exists(filepath_ir_):
                # Check if segmentation label exists
                filepath_label_ = os.path.join(data_dir_labels, file)
                if os.path.exists(filepath_label_):
                    self.filepath_vis.append(filepath_vis_)
                    self.filenames_vis.append(file)
                    self.filepath_ir.append(filepath_ir_)
                    self.filenames_ir.append(file)
                    self.filepath_labels.append(filepath_label_)
                    self.filenames_labels.append(file)
        
        print(f"Successfully loaded {len(self.filepath_vis)} valid image pairs for MFNet")
        
        # Update length to actual number of valid pairs
        if self.length == 0 or self.length > len(self.filepath_vis):
            self.length = len(self.filepath_vis)

    def _load_pst900_data(self):
        """Load PST900 dataset, only add valid pairs"""
        data_dir_base = "/data/harris/FusionMamba/PST900/PST900_RGBT_Dataset/train"
        data_dir_vis = os.path.join(data_dir_base, "rgb")      # RGB images
        data_dir_ir = os.path.join(data_dir_base, "thermal")   # Thermal images
        data_dir_labels = os.path.join(data_dir_base, "labels") # Segmentation labels
        vis_files = [f for f in os.listdir(data_dir_vis) if f.endswith(('.jpg', '.png', '.bmp', '.tif'))]
        vis_files.sort()
        valid_count = 0
        for file in vis_files:
            filepath_vis_ = os.path.join(data_dir_vis, file)
            filepath_ir_ = os.path.join(data_dir_ir, file)
            if os.path.exists(filepath_ir_):
                # Check if segmentation label exists
                filepath_label_ = os.path.join(data_dir_labels, file)
                if os.path.exists(filepath_label_):
                    # Try loading both images to ensure they're not corrupted
                    vis_img = cv2.imread(filepath_vis_)
                    ir_img = cv2.imread(filepath_ir_, 0)
                    label_img = cv2.imread(filepath_label_, 0)
                    if vis_img is not None and ir_img is not None and label_img is not None:
                        self.filepath_vis.append(filepath_vis_)
                        self.filenames_vis.append(file)
                        self.filepath_ir.append(filepath_ir_)
                        self.filenames_ir.append(file)
                        self.filepath_labels.append(filepath_label_)
                        self.filenames_labels.append(file)
                        valid_count += 1
        print(f"Successfully loaded {valid_count} valid image pairs for PST900")
        if self.length == 0 or self.length > len(self.filepath_vis):
            self.length = len(self.filepath_vis)

    def _load_pst900_test_data(self):
        """Load PST900 test dataset, only add valid pairs"""
        data_dir_base = "/data/harris/FusionMamba/PST900/PST900_RGBT_Dataset/test"
        data_dir_vis = os.path.join(data_dir_base, "rgb")      # RGB images
        data_dir_ir = os.path.join(data_dir_base, "thermal")   # Thermal images
        data_dir_labels = os.path.join(data_dir_base, "labels") # Segmentation labels
        vis_files = [f for f in os.listdir(data_dir_vis) if f.endswith(('.jpg', '.png', '.bmp', '.tif'))]
        vis_files.sort()
        valid_count = 0
        for file in vis_files:
            filepath_vis_ = os.path.join(data_dir_vis, file)
            filepath_ir_ = os.path.join(data_dir_ir, file)
            if os.path.exists(filepath_ir_):
                # Check if segmentation label exists
                filepath_label_ = os.path.join(data_dir_labels, file)
                if os.path.exists(filepath_label_):
                    # Try loading both images to ensure they're not corrupted
                    vis_img = cv2.imread(filepath_vis_)
                    ir_img = cv2.imread(filepath_ir_, 0)
                    label_img = cv2.imread(filepath_label_, 0)
                    if vis_img is not None and ir_img is not None and label_img is not None:
                        self.filepath_vis.append(filepath_vis_)
                        self.filenames_vis.append(file)
                        self.filepath_ir.append(filepath_ir_)
                        self.filenames_ir.append(file)
                        self.filepath_labels.append(filepath_label_)
                        self.filenames_labels.append(file)
                        valid_count += 1
        print(f"Successfully loaded {valid_count} valid test image pairs for PST900")
        if self.length == 0 or self.length > len(self.filepath_vis):
            self.length = len(self.filepath_vis)

    def _load_mfnet_test_data(self):
        """Load MFNet test dataset, only add valid pairs"""
        data_dir_base = "/data/harris/FusionMamba/MFNet/ir_seg_dataset"
        data_dir_vis = os.path.join(data_dir_base, "images")  # RGB images
        data_dir_ir = os.path.join(data_dir_base, "visual")   # Thermal images
        data_dir_labels = os.path.join(data_dir_base, "labels") # Segmentation labels
        
        # Read test file list
        test_file = os.path.join(data_dir_base, "test.txt")
        with open(test_file, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        
        valid_count = 0
        for file_id in test_files:
            # MFNet uses specific naming convention
            rgb_file = f"{file_id}.png"
            thermal_file = f"{file_id}.jpg"
            label_file = f"{file_id}.png"
            
            filepath_vis_ = os.path.join(data_dir_vis, rgb_file)
            filepath_ir_ = os.path.join(data_dir_ir, thermal_file)
            filepath_label_ = os.path.join(data_dir_labels, label_file)
            
            # Check if all files exist
            if os.path.exists(filepath_vis_) and os.path.exists(filepath_ir_) and os.path.exists(filepath_label_):
                # Try loading all images to ensure they're not corrupted
                vis_img = cv2.imread(filepath_vis_)
                ir_img = cv2.imread(filepath_ir_, 0)
                label_img = cv2.imread(filepath_label_, 0)
                if vis_img is not None and ir_img is not None and label_img is not None:
                    self.filepath_vis.append(filepath_vis_)
                    self.filenames_vis.append(rgb_file)
                    self.filepath_ir.append(filepath_ir_)
                    self.filenames_ir.append(thermal_file)
                    self.filepath_labels.append(filepath_label_)
                    self.filenames_labels.append(label_file)
                    valid_count += 1
        
        print(f"Successfully loaded {valid_count} valid test image pairs for MFNet")
        if self.length == 0 or self.length > len(self.filepath_vis):
            self.length = len(self.filepath_vis)

    def __getitem__(self, index):
        # Robustly skip bad files
        max_attempts = len(self.filepath_vis)
        attempts = 0
        while attempts < max_attempts:
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = cv2.imread(vis_path)
            image_ir = cv2.imread(ir_path, 0)
            if image_vis is None or image_ir is None:
                # Skip to next index (wrap around)
                index = (index + 1) % len(self.filepath_vis)
                attempts += 1
                continue
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            image_ir, image_vis = self.resize(image_ir, image_vis, [256, 256], [256, 256])
            image_vis_y = cv2.cvtColor(image_vis, cv2.COLOR_RGB2GRAY)
            image_vis_y = np.asarray(Image.fromarray(image_vis_y), dtype=np.float32) / 255.0
            image_vis_y = np.expand_dims(image_vis_y, axis=0)
            image_ir = np.asarray(Image.fromarray(image_ir), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            
            # Load and process segmentation label if available
            if hasattr(self, 'filepath_labels') and len(self.filepath_labels) > 0:
                label_path = self.filepath_labels[index]
                label_img = cv2.imread(label_path, 0)
                if label_img is not None:
                    # Resize labels with nearest neighbor to preserve class values
                    label_img = imresize(label_img, [256, 256], interp='nearest')
                    label_img = np.asarray(Image.fromarray(label_img), dtype=np.float32)
                    
                    # Convert labels for different datasets
                    if self.dataset_name == 'acod':
                        # For ACOD (binary segmentation): convert [0,255] to [0,1]
                        label_img = (label_img > 127).astype(np.float32)
                    else:
                        # For MFNet/PST900 (multi-class): labels should already be in [0, num_classes-1]
                        # Ensure labels are integers and within valid range
                        label_img = np.round(label_img).astype(np.float32)
                        label_img = np.clip(label_img, 0, 8)  # Clip to valid class range [0, 8]
                    
                    return (
                        torch.tensor(image_vis_y),
                        torch.tensor(image_ir),
                        torch.tensor(label_img, dtype=torch.long),
                    )
            
            return (
                torch.tensor(image_vis_y),
                torch.tensor(image_ir),
            )
        # If all attempts fail, raise error
        raise ValueError(f"All attempts to load images failed for dataset {self.dataset_name}")

    def __len__(self):
        return self.length

    def resize(self, data, data2, crop_size_img, crop_size_label):
        data = imresize(data, crop_size_img, interp='bicubic')
        data2 = imresize(data2, crop_size_label, interp='bicubic')
        return data, data2
