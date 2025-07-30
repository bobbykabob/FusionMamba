import torch
import numpy as np
import os
from models.vmamba_Fusion_efficross import VSSM_Fusion_Segmentation
from TaskFusion_dataset import Fusion_dataset


class runningScore(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls1 = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls1)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum())
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_acc = dict(zip(range(self.n_classes), acc_cls1))
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "pixel_acc: ": acc,
                "class_acc: ": acc_cls,
                "mIou: ": mean_iou,
                "fwIou: ": fw_iou,
            },

            cls_acc,
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_pst900_segmentation_proper():
    """Evaluate PST900 segmentation using proper pytorch-semseg metrics"""
    
    print("=" * 70)
    print("PST900 Segmentation Evaluation with Proper pytorch-semseg Metrics")
    print("=" * 70)
    
    # Load model
    model_path = "model_last/segmentation/segmentation_model_pst900.pth"
    print(f"Loading segmentation model from: {model_path}")
    
    model = VSSM_Fusion_Segmentation(num_seg_classes=5)  # PST900 has 5 classes
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    print("Segmentation model loaded successfully!")
    
    # Load test dataset
    test_dataset = Fusion_dataset('test', dataset_name='pst900')
    print(f"Successfully loaded {len(test_dataset)} valid test image pairs for PST900")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize metrics
    running_metrics = runningScore(n_classes=5)  # PST900 has 5 classes
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            data = test_dataset[i]
            image_vis, image_ir, target = data
            
            # Add batch dimension
            image_vis = image_vis.unsqueeze(0).cuda()
            image_ir = image_ir.unsqueeze(0).cuda()
            target = target.cuda()
            
            # Forward pass
            seg_logits = model.forward_segmentation_only(image_vis, image_ir)
            pred = torch.argmax(seg_logits, dim=1)
            
            # Convert to numpy for metrics
            pred_np = pred.cpu().numpy().squeeze()
            target_np = target.cpu().numpy().squeeze()
            
            # Update metrics
            running_metrics.update([target_np], [pred_np])
            
            # Print progress
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")
    
    # Get final scores
    score, class_acc, class_iou = running_metrics.get_scores()
    
    print("\n" + "=" * 70)
    print("PST900 Segmentation Evaluation Results (Proper Metrics):")
    print("=" * 70)
    print(f"Pixel Accuracy: {score['pixel_acc: ']:.4f}")
    print(f"Mean Class Accuracy (mAcc): {score['class_acc: ']:.4f}")
    print(f"Mean IoU (mIoU): {score['mIou: ']:.4f}")
    print(f"Frequency Weighted IoU: {score['fwIou: ']:.4f}")
    print(f"Number of test samples: {len(test_dataset)}")
    print("=" * 70)
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    class_names = {0: 'Background', 1: 'Person', 2: 'Car', 3: 'Bicycle', 4: 'Motorcycle'}
    print(f"{'Class':<12} {'IoU':<8} {'Accuracy':<10}")
    print("-" * 70)
    for class_id in range(5):
        class_name = class_names[class_id]
        iou = class_iou[class_id]
        acc = class_acc[class_id]
        if not np.isnan(iou):
            print(f"{class_name:<12} {iou:<8.4f} {acc:<10.4f}")
        else:
            print(f"{class_name:<12} {'N/A':<8} {'N/A':<10}")
    
    # Save results
    with open("pst900_segmentation_results_proper.txt", "w") as f:
        f.write("PST900 Segmentation Evaluation Results (Proper Metrics):\n")
        f.write("=" * 70 + "\n")
        f.write(f"Pixel Accuracy: {score['pixel_acc: ']:.4f}\n")
        f.write(f"Mean Class Accuracy (mAcc): {score['class_acc: ']:.4f}\n")
        f.write(f"Mean IoU (mIoU): {score['mIou: ']:.4f}\n")
        f.write(f"Frequency Weighted IoU: {score['fwIou: ']:.4f}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model used: {model_path}\n")
        f.write("Note: Using proper pytorch-semseg metrics calculation.\n")
    
    print(f"\nResults saved to pst900_segmentation_results_proper.txt")


if __name__ == "__main__":
    evaluate_pst900_segmentation_proper() 