import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DoubleDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        """
        PyTorch Shape: [Batch, Channel, H, W]
        TF Original: axis=[1,2] -> H, W (Channel is last in TF)
        PyTorch Equivalent: Sum over [2, 3] if input is [B, C, H, W]
        """
        # 確保輸入在 0~1 之間 (如果模型最後沒有 Sigmoid，這裡要加)
        # y_pred = torch.sigmoid(y_pred) 
        
        # 展平 H, W 維度以便計算
        # [B, C, H, W] -> [B, C, H*W]
        batch, channel, h, w = y_true.shape
        y_true_f = y_true.view(batch, channel, -1)
        y_pred_f = y_pred.view(batch, channel, -1)

        # Term 1: Foreground Dice
        intersection = torch.sum(y_true_f * y_pred_f, dim=2)
        union = torch.sum(y_true_f + y_pred_f, dim=2)
        term1 = (intersection + self.smooth) / (union + self.smooth)

        # Term 2: Background Dice (1 - y)
        intersection_inv = torch.sum((1 - y_true_f) * (1 - y_pred_f), dim=2)
        union_inv = torch.sum((2 - y_true_f - y_pred_f), dim=2)
        term2 = (intersection_inv + self.smooth) / (union_inv + self.smooth)

        # Final Loss: 1 - Term1 - Term2
        # 注意：通常 Dice Loss 是 1 - Dice。這裡是 1 - FG_Dice - BG_Dice
        # 但這可能導致負值 (因為 Dice 最大是 1)，學長原公式是這樣寫的，我們先照搬。
        # 原公式：1 - (FG) - (BG)。如果 FG=1, BG=1，Loss = -1。
        # 建議檢查一下是否應該是 1 - (FG + BG)/2 或者是 2 - FG - BG
        # 為了忠實復刻，這裡保持原樣：
        return 1 - term1 - term2

class WeightedConsecutiveLoss(nn.Module):
    def __init__(self, loss_ratio=0.5, consecutive=3, dynamic_weight=False):
        super(WeightedConsecutiveLoss, self).__init__()
        self.loss_ratio = loss_ratio
        self.consecutive = consecutive
        self.dynamic_weight = dynamic_weight
        self.base_loss = DoubleDiceLoss()

    def forward(self, y_true, y_pred):
        # 假設 y_true, y_pred 形狀為 [Batch, C, H, W]
        
        # 1. 固定權重模式 (Fixed Ratio)
        if not self.dynamic_weight and self.loss_ratio != 0:
            if self.consecutive == 2:
                # 權重計算
                current_ratio = self.loss_ratio / (self.loss_ratio + 1)
                neighbor_ratio = 1 / (self.loss_ratio + 1)
                
                print(f"Ratio: Curr={current_ratio:.2f}, Prev={neighbor_ratio:.2f}")

                loss_center = current_ratio * self.base_loss(y_true, y_pred)
                
                # Cross-Temporal Loss: True[t] vs Pred[t+1]
                # 注意：這裡會讓 Batch Size 變少 1
                loss_prev = neighbor_ratio * self.base_loss(y_true[:-1], y_pred[1:])
                
                # 為了將 loss 合併，我們取 mean
                return torch.mean(loss_center) + torch.mean(loss_prev)

            else: # consecutive == 3 (CRNN Default)
                # 權重計算
                total_share = self.loss_ratio * 2 + 2
                current_ratio = (self.loss_ratio * 2) / total_share
                neighbor_ratio = 1 / total_share

                # loss_center: [Batch]
                loss_center = current_ratio * self.base_loss(y_true, y_pred)
                
                # loss_left: True[t] vs Pred[t+1] (Shifted Left)
                loss_left = neighbor_ratio * self.base_loss(y_true[:-1], y_pred[1:])
                
                # loss_right: True[t] vs Pred[t-1] (Shifted Right)
                loss_right = neighbor_ratio * self.base_loss(y_true[1:], y_pred[:-1])

                # 原代碼使用了 tf.concat 然後 mean。
                # 在 PyTorch 直接加總 mean 即可，數學上等價
                final_loss = torch.mean(loss_center) + torch.mean(loss_left) + torch.mean(loss_right)
                return final_loss

        # 2. 動態權重模式 (Dynamic Weight) - 只有 consecutive=3
        elif self.dynamic_weight or self.loss_ratio == 0:
            if self.consecutive != 3:
                raise ValueError("Dynamic weight only supports consecutive=3")
            
            # --- 計算動態權重 (Dynamic Logic) ---
            # TF: Y = K.concatenate([Y[0], Y, Y[-1]]) padding batch
            # 這是為了計算每一幀跟前後幀的差異
            y_pad = torch.cat([y_true[0:1], y_true, y_true[-1:]], dim=0)
            
            # 差異計算: |Y[t+1] - Y[t]|
            # Sum over (C, H, W) -> dims [1, 2, 3]
            diff = torch.abs(y_pad[1:] - y_pad[:-1])
            prev_curr_diff = torch.sum(diff, dim=[1, 2, 3]) # [Batch+1]
            
            # 歸一化差異
            y_sum = torch.sum(y_true, dim=[1, 2, 3]) # [Batch]
            # y_pad[:-1] 對應的是原長度的 padding 版本，這裡簡化邏輯：
            # 學長原代碼：prevCurrDiff / sum(Y[:-1]) * 120
            # 為了對齊長度，我們取 diff 的前 N 個
            diff_metric = (prev_curr_diff[:-1] / (y_sum + 1e-6)) * 120
            
            # 處理 NaN
            diff_metric = torch.nan_to_num(diff_metric, 0.0)
            
            const = 2.0
            # mask: if diff_metric == 0
            mask = (diff_metric == 0).float()
            center_weight_raw = (const - 1) * mask + diff_metric
            
            # 建構三項權重
            # Batch Weight: [Left, Center, Right]
            # Center = diff[t] * diff[t-1]
            cw_curr = center_weight_raw[1:]
            cw_prev = center_weight_raw[:-1]
            
            w_center = cw_curr * cw_prev
            w_left = cw_curr
            w_right = cw_prev
            
            denominator = w_center + w_left + w_right + 1e-6
            
            # 最終 Ratio [Batch-1] (因為 diff 會少一個維度)
            # 注意：這裡維度對齊非常痛苦，建議如果是初期復刻，
            # 強烈建議先用 Fixed Ratio (loss_ratio != 0)。
            # Dynamic Weight 很容易因為 Batch Size 太小而算出一堆 NaN。
            
            ratio_center = w_center / denominator
            ratio_left = w_left / denominator
            ratio_right = w_right / denominator
            
            # 計算 Loss
            # 因為使用了 padding 和 shifting，這裡的 batch size 會縮減
            # 我們需要對齊 loss 和 ratio 的長度
            
            # 取中間段的 loss
            l_c = self.base_loss(y_true[1:-1], y_pred[1:-1])
            l_l = self.base_loss(y_true[:-2], y_pred[1:-1]) # y_true[t-1] vs y_pred[t]
            l_r = self.base_loss(y_true[2:], y_pred[1:-1])  # y_true[t+1] vs y_pred[t]
            
            # 確保 ratio 長度跟 loss 一樣 (Batch - 2)
            # 這裡做一個簡單的 slice
            valid_len = l_c.shape[0]
            r_c = ratio_center[:valid_len].view(-1, 1) # view 用於廣播
            r_l = ratio_left[:valid_len].view(-1, 1)
            r_r = ratio_right[:valid_len].view(-1, 1)

            final_loss = torch.mean(r_c * l_c + r_l * l_l + r_r * l_r)
            return final_loss

        else:
            return self.base_loss(y_true, y_pred)