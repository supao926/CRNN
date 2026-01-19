import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
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
        skip = x 
        out = self.pool(x)
        out = self.dropout(out)
        return out, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout_rate):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn_up = nn.BatchNorm2d(out_ch)
        
        # 這裡假設 skip connection 來自多幀融合，通道數可能會增加
        # 但在 decoder 我們假設 skip 通道數是固定的
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + 2 * skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_curr, skip_prev):
        x = self.up(x)
        x = self.bn_up(x)
        
        if x.size() != skip_curr.size():
            x = F.interpolate(x, size=skip_curr.shape[2:], mode='bilinear', align_corners=True)

        cat_x = torch.cat([skip_curr, skip_prev, x], dim=1)
        return self.conv(cat_x)

class ConcurrentUNet(nn.Module):
    def __init__(self, config): # config 來自 config.py 或 dict
        super().__init__()
        self.n_frames = config.N_FRAMES
        self.dropout_rate = config.DROPOUT_RATE
        
        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        in_ch = getattr(config, 'input_channels', 1) # 動態取得
        
        for out_ch in config.BLOCK_CHANNELS:
            self.encoder_blocks.append(
                ConvBlock(in_ch, out_ch, 2, self.dropout_rate)
            )
            in_ch = out_ch 
            
        # --- Bottleneck ---
        bottleneck_in_ch = config.BLOCK_CHANNELS[-1] * self.n_frames 
        bottleneck_out_ch = config.BLOCK_CHANNELS[-1]
        
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_in_ch, bottleneck_out_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        rev_channels = config.BLOCK_CHANNELS[::-1] 
        
        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i]
            skip_ch = rev_channels[i+1]
            out_ch = rev_channels[i+1]
            self.decoder_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, self.dropout_rate)
            )
            
        self.final_conv = nn.Conv2d(config.BLOCK_CHANNELS[0], config.NUM_CLASSES, kernel_size=1)

    def forward_single_branch(self, x):
        skips = []
        for block in self.encoder_blocks:
            x, skip = block(x)
            skips.append(skip)
        return x, skips

    def forward(self, x_prev, x_curr, x_next=None):
        feat_prev, skips_prev = self.forward_single_branch(x_prev)
        feat_curr, skips_curr = self.forward_single_branch(x_curr)
        
        feats = [feat_prev, feat_curr]
        if self.n_frames == 3:
            if x_next is None:
                 raise ValueError("n_frames=3 but x_next is None")
            feat_next, _ = self.forward_single_branch(x_next)
            feats.append(feat_next)
        
        # Bottleneck 融合
        bottleneck = torch.cat(feats, dim=1) 
        x = self.bottleneck_conv(bottleneck)
        
        # Decoder
        for i, up_block in enumerate(self.decoder_blocks):
            idx = -(i + 2) 
            s_curr = skips_curr[idx]
            s_prev = skips_prev[idx]
            x = up_block(x, s_curr, s_prev)
            
        logits = self.final_conv(x)
        return logits