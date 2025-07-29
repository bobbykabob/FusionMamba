import numpy as np

def macc_present_only(pred, target, num_classes=9):
    """Only average over classes present in GT (our current approach)"""
    accuracies = []
    for class_id in range(num_classes):
        pred_binary = (pred == class_id)
        target_binary = (target == class_id)
        if np.sum(target_binary) > 0:
            accuracy = np.sum(pred_binary & target_binary) / np.sum(target_binary)
            accuracies.append(accuracy)
    return np.mean(accuracies) if accuracies else 0.0

def macc_all_classes(pred, target, num_classes=9):
    """Average over all classes, zero for missing ones"""
    accuracies = []
    for class_id in range(num_classes):
        pred_binary = (pred == class_id)
        target_binary = (target == class_id)
        if np.sum(target_binary) > 0:
            accuracy = np.sum(pred_binary & target_binary) / np.sum(target_binary)
        else:
            accuracy = 0.0
        accuracies.append(accuracy)
    return np.mean(accuracies)

# Test with realistic MFNet data
pred = np.zeros(256*256)
target = np.zeros(256*256)

# Set some classes present, some missing
pred[0:60000] = 0  # Background
pred[60000:61000] = 1  # Person
pred[61000:62000] = 2  # Car
target[0:60000] = 0
target[60000:61000] = 1
target[61000:62000] = 2

print('Test with 3 classes present, 6 missing:')
print(f'mAcc (present only): {macc_present_only(pred, target):.4f}')
print(f'mAcc (all classes): {macc_all_classes(pred, target):.4f}')

# Test with perfect predictions
pred = target.copy()
print(f'\nTest with perfect predictions:')
print(f'mAcc (present only): {macc_present_only(pred, target):.4f}')
print(f'mAcc (all classes): {macc_all_classes(pred, target):.4f}') 