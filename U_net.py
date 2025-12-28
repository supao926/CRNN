import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

"""
# 1. 配置管理 (取代 argparse)
@dataclass
class ModelConfig:
    block_channels: list = (64, 128, 256, 512) # 範例深度
    block_conv: int = 2        # 每個 Block 有幾層 Conv
    dropout_rate: float = 0.25
    n_frames: int = 3          # n=2 或 n=3 (對應 prev, curr, next)
    input_channels: int = 1    # 灰階為 1，RGB 為 3
    num_classes: int = 2       # 最終輸出的類別數 (對應 output_layer)
"""

# 2. 基礎卷積塊 (取代 set_or_get_conv2d)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 第一層負責改變通道數，後續層維持通道數
            cin = in_ch if i == 0 else out_ch
            self.layers.append(nn.Sequential(
                nn.Conv2d(cin, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        skip = x  # 保存用於 Skip Connection 的特徵 (Before Pooling)
        out = self.pool(x)
        out = self.dropout(out)
        return out, skip

# 3. 上採樣塊 (取代 Decoder 中的重複邏輯)
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout_rate):
        super().__init__()
        # 對應 Conv2DTranspose(strides=2)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn_up = nn.BatchNorm2d(out_ch)
        
        # 融合後的卷積 (Skip connection channel size + Upsampled channel size)
        # 根據舊代碼：concatenate([curr_skip, prev_skip, up]) -> 2 * skip_ch + out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + 2 * skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_curr, skip_prev):
        x = self.up(x)
        x = self.bn_up(x)
        
        # 確保維度匹配 (處理 padding 問題)
        if x.size() != skip_curr.size():
            x = F.interpolate(x, size=skip_curr.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate: [Curr, Prev, Up]
        cat_x = torch.cat([skip_curr, skip_prev, x], dim=1)
        return self.conv(cat_x)

from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    # 1. 輸入與輸出規格
    input_channels: int = 1      # 輸入圖片的通道數 (醫學影像通常是 1，若是 RGB 則是 3)
    num_classes: int = 1         # 最終輸出的類別數 (若是做二元分割如：肝臟/背景，則為 1)
    
    # 2. 模型深度與寬度
    # 這決定了 U-Net 有幾層，以及每層變多厚
    # [64, 128, 256, 512] 代表有 4 個 Encoder Block，最底層是 512
    block_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # 3. 區塊內部的複雜度
    block_conv: int = 2          # 每個 ConvBlock 裡面重複做幾次卷積 (num_layers)
    
    # 4. 時序相關 (Concurrent 的核心)
    n_frames: int = 3            # 一次輸入幾張圖？(2 or 3) 這會影響 Bottleneck 的輸入厚度
    
    # 5. 正則化參數
    dropout_rate: float = 0.5    # Dropout 的比例

# 4. 主模型：Concurrent CRNN (Siamese U-Net)
class ConcurrentUNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # --- Encoder (權重共享) ---
        self.encoder_blocks = nn.ModuleList()
        in_ch = config.input_channels
        
        for out_ch in config.block_channels:
            self.encoder_blocks.append(
                ConvBlock(in_ch, out_ch, config.block_conv, config.dropout_rate)
            )
            in_ch = out_ch # 更新下一層的輸入
            
        # --- Bottleneck (最底層) ---
        # 根據舊代碼：如果是 n=3，concatenate(prev, curr, next)
        # 輸入通道數 = 最後一層 Encoder Channel * n_frames
        bottleneck_in_ch = config.block_channels[-1] * config.n_frames 
        bottleneck_out_ch = config.block_channels[-1]
        
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_in_ch, bottleneck_out_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate)
        )

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # 倒序遍歷 channels (排除最後一層，因為那是 bottleneck 的輸入)
        rev_channels = config.block_channels[::-1] 
        
        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i]
            skip_ch = rev_channels[i+1] # 下一層的 channel (因為是倒序)
            out_ch = rev_channels[i+1]
            
            self.decoder_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, config.dropout_rate)
            )
            
        # 最終輸出層
        self.final_conv = nn.Conv2d(config.block_channels[0], config.num_classes, kernel_size=1)

    def forward_single_branch(self, x):
        """跑單一分支的 Encoder"""
        skips = []
        for block in self.encoder_blocks:
            x, skip = block(x)
            skips.append(skip)
        return x, skips

    def forward(self, x_prev, x_curr, x_next=None):
        # 1. Encoder 階段 (權重共享：同一個 self.encoder_blocks 跑多次)
        feat_prev, skips_prev = self.forward_single_branch(x_prev)
        feat_curr, skips_curr = self.forward_single_branch(x_curr)
        
        feats = [feat_prev, feat_curr]
        if self.config.n_frames == 3 and x_next is not None:
            feat_next, _ = self.forward_single_branch(x_next)
            feats.append(feat_next) # [Prev, Curr, Next]
        
        # 2. Bottleneck 融合
        # 舊代碼邏輯：concatenate([prev_last, curr_last, (next_last)])
        bottleneck = torch.cat(feats, dim=1) 
        x = self.bottleneck_conv(bottleneck)
        
        # 3. Decoder 階段
        # 需要倒序取出 skips (因為 Decoder 是從底層往上)
        # skips 列表是 [Block0, Block1, Block2, Block3(Bottleneck input)]
        # 我們需要從 Block2 開始往回拿
        
        for i, up_block in enumerate(self.decoder_blocks):
            # 取得對應層級的 skip connection
            # i=0 時，處理的是倒數第二層的特徵
            idx = -(i + 2) 
            s_curr = skips_curr[idx]
            s_prev = skips_prev[idx]
            
            x = up_block(x, s_curr, s_prev)
            
        # 4. 輸出
        logits = self.final_conv(x)
        return logits

# --- 測試區 ---
if __name__ == '__main__':
    # 模擬輸入 (Batch=1, Channel=1, H=256, W=256)
    dummy_prev = torch.randn(1, 1, 256, 256)
    dummy_curr = torch.randn(1, 1, 256, 256)
    dummy_next = torch.randn(1, 1, 256, 256)

    # 初始化配置與模型
    cfg = ModelConfig(n_frames=3, block_channels=[32, 64, 128])
    model = ConcurrentUNet(cfg)
    
    # 執行 Forward Pass
    print("Model initialized. Running forward pass...")
    output = model(dummy_prev, dummy_curr, dummy_next)
    print("Output shape:", output.shape) # 預期: [1, 2, 256, 256]