import torch
import numpy as np

def calculate_metrics(pred_logits, target, threshold=0.5):
    """
    pred_logits: 未經 Sigmoid 的輸出
    target: Ground Truth (0 or 1)
    """
    pred_probs = torch.sigmoid(pred_logits)
    pred_bin = (pred_probs > threshold).float()
    
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    eps = 1e-6
    dice = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    
    correct = (pred_flat == target_flat).float().sum()
    total = torch.tensor(target_flat.numel()).float().to(pred_flat.device)
    acc = correct / total
    
    return {"dice": dice.item(), "om": iou.item(), "acc": acc.item()}

class VolumeMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0
        self.pred_sum = 0
        self.gt_sum = 0
        self.fp_count = 0

    def update(self, pred, gt):
        if hasattr(pred, 'cpu'): pred = pred.cpu().numpy()
        if hasattr(gt, 'cpu'): gt = gt.cpu().numpy()
        
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)

        self.intersection += np.sum(pred * gt)
        self.pred_sum += np.sum(pred)
        self.gt_sum += np.sum(gt)
        self.fp_count += np.sum(pred * (1 - gt)) 
        
    def get_metrics(self):
        epsilon = 1e-6
        om = self.intersection / (self.pred_sum + self.gt_sum - self.intersection + epsilon)
        tp = self.intersection / (self.gt_sum + epsilon)
        fp = self.fp_count / (self.gt_sum + epsilon)
        dr = 2 * self.intersection / (self.pred_sum + self.gt_sum + epsilon)

        return {"OM": om, "TP": tp, "FP": fp, "Dice": dr}