import numpy as np
import pandas as pd

import torch
import numpy as np

def calculate_metrics(pred_logits, target, threshold=0.5):
    """
    一次計算所有指標：Dice, IoU (OM), Accuracy
    pred_logits: (B, 1, H, W) 未經過 Sigmoid 的輸出
    target: (B, 1, H, W) 0 或 1
    """
    # 1. 前處理
    pred_probs = torch.sigmoid(pred_logits)
    pred_bin = (pred_probs > threshold).float()
    
    # 為了計算方便，展平
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    # 2. 核心數值
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    # epsilon 防止除以 0
    eps = 1e-6
    
    # 3. 計算指標
    # Dice Score
    dice = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    
    # IoU (Overlap Metric - OM)
    # Union = A + B - Inter
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    
    # Accuracy (Pixel-wise)
    # 正確預測的像素數 / 總像素數
    correct = (pred_flat == target_flat).float().sum()
    total = torch.tensor(target_flat.numel()).float().to(pred_flat.device)
    acc = correct / total
    
    return {
        "dice": dice.item(),
        "om": iou.item(),  # 學長的 OM 就是 IoU
        "acc": acc.item()
    }

class VolumeMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0
        self.pred_sum = 0
        self.gt_sum = 0
        self.fp_count = 0
        self.tn_count = 0

    def update(self, pred, gt):
        """
        輸入單張或多張 Slice 的預測結果與 GT
        pred: (B, H, W) 0 or 1 (Binary)
        gt:   (B, H, W) 0 or 1 (Binary)
        """
        # 轉成 numpy 並確保是 boolean/int
        if hasattr(pred, 'cpu'): pred = pred.cpu().numpy()
        if hasattr(gt, 'cpu'): gt = gt.cpu().numpy()
        
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)

        # 核心計算 (與學長邏輯一致)
        self.intersection += np.sum(pred * gt)
        self.pred_sum += np.sum(pred)
        self.gt_sum += np.sum(gt)
        
        # 用於計算 FP (學長定義: FP = (Pred總和 / GT總和) - Sensitivity)
        # 或者傳統定義: FP pixels = pred & (1-gt)
        self.fp_count += np.sum(pred * (1 - gt)) 
        
    def get_metrics(self):
        """回傳計算好的字典"""
        # 避免除以 0
        epsilon = 1e-6
        
        # 1. Overlap Metric (OM) / IoU
        # 學長公式: inter / (pred + gt - inter)
        om = self.intersection / (self.pred_sum + self.gt_sum - self.intersection + epsilon)
        
        # 2. True Positive Rate (TP) / Sensitivity
        # 學長公式: inter / gt
        tp = self.intersection / (self.gt_sum + epsilon)
        
        # 3. False Positive Rate (FP) - Ansen's definition
        # 學長公式: (pred_sum / gt_sum) - sensitivity
        # 數學上等價於: (pred_sum - intersection) / gt_sum = fp_pixels / gt_sum
        fp = self.fp_count / (self.gt_sum + epsilon)
        
        # 4. Dice Score (DR)
        # 學長公式: 2 * inter / (pred + gt)
        dr = 2 * self.intersection / (self.pred_sum + self.gt_sum + epsilon)

        return {
            "OM": om,
            "TP": tp,
            "FP": fp,
            "Dice": dr,
            "GT_Vol": self.gt_sum,
            "Pred_Vol": self.pred_sum
        }