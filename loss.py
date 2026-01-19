import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    【本次唯一的改動點】
    原本學長的 Loss: 包含背景項 + 少了係數 2 -> 導致全黑
    修正後的 Loss: 移除背景項 + 補回係數 2 -> 專注於腫瘤
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        # 輸入形狀: [Batch, Channel, H, W]
        batch, channel, h, w = y_true.shape
        y_true_f = y_true.view(batch, channel, -1)
        y_pred_f = y_pred.view(batch, channel, -1)

        # 改動 1: 只計算前景 (移除 term2 背景項)
        intersection = torch.sum(y_true_f * y_pred_f, dim=2)
        union = torch.sum(y_true_f, dim=2) + torch.sum(y_pred_f, dim=2)
        
        # 改動 2: 補上係數 2.0
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - torch.mean(dice_score)

class WeightedConsecutiveLoss(nn.Module):
    def __init__(self, loss_ratio=0.5, consecutive=3): 
        #這裡我保持 loss_ratio=0.5 (學長原設)，不變動此變因
        super(WeightedConsecutiveLoss, self).__init__()
        self.consecutive = consecutive
        self.base_loss = DiceLoss() 
        
        # 學長的原始權重公式 (保持不動)
        total_share = loss_ratio * 2 + 2
        self.w_center = (loss_ratio * 2) / total_share
        self.w_side = 1.0 / total_share
        
        print(f"Experimental Control: Ratio={loss_ratio}, Center={self.w_center:.2f}, Side={self.w_side:.2f}")

    def forward(self, y_true, y_pred):
        # 邏輯架構保持不動，只改變底層的 self.base_loss
        loss_center = self.base_loss(y_true, y_pred)
        
        if y_true.size(0) < 2: return loss_center

        if self.consecutive == 3:
            loss_left = self.base_loss(y_true[:-1], y_pred[1:])
            loss_right = self.base_loss(y_true[1:], y_pred[:-1])
            
            return (self.w_center * loss_center) + \
                   (self.w_side * loss_left) + \
                   (self.w_side * loss_right)

        elif self.consecutive == 2:
            loss_prev = self.base_loss(y_true[:-1], y_pred[1:])
            w_total = self.w_center + (self.w_side * 2)
            return (self.w_center/w_total * loss_center) + ((self.w_side*2)/w_total * loss_prev)
            
        else:
            return loss_center