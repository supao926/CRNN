import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

# 匯入專案模組
from config import Config
from dataset import CRNNDataset
from model import ConcurrentUNet, ModelConfig
from loss import WeightedConsecutiveLoss
from utils import calculate_metrics

def train():
    # --- 1. 初始化與配置 ---
    cfg = Config()
    
    # 建立存檔目錄
    save_dir = cfg.CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # 記錄檔 (對應學長的 logs.json)
    log_file = os.path.join(save_dir, 'training_log.json')
    
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Start Training on {device}...")

    # --- 2. 資料準備 ---
    # 這裡我們將資料集切分為 Train (80%) 和 Validation (20%)
    # 學長是用 split_num 手動切，這裡用 PyTorch 標準做法
    full_ds = CRNNDataset(
        root_dir=cfg.TRAIN_IMG_DIR,
        target_dir=cfg.TRAIN_MASK_DIR,
        n_frames=cfg.N_FRAMES,
        use_blur=cfg.USE_BLUR,
        use_grad=cfg.USE_GRAD
    )
    
    # 自動偵測 channel 數
    sample_ch = full_ds[0]['curr'].shape[0]
    print(f"Detected Input Channels: {sample_ch}")
    
    val_size = int(len(full_ds) * 0.2)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    print(f"Data Split: Train={len(train_ds)}, Val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. 模型與優化器 ---
    model_cfg = ModelConfig(
        block_channels=cfg.BLOCK_CHANNELS,
        dropout_rate=cfg.DROPOUT_RATE,
        n_frames=cfg.N_FRAMES,
        input_channels=sample_ch,
        num_classes=cfg.NUM_CLASSES
    )
    model = ConcurrentUNet(model_cfg).to(device)

    criterion = WeightedConsecutiveLoss(loss_ratio=0.5, consecutive=cfg.N_FRAMES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # --- 4. 監控變數 (還原學長的 Checkpoint 邏輯) ---
    best_metrics = {
        'val_loss': float('inf'), # Min
        'val_om': float('-inf'),  # Max
        'train_loss': float('inf'),
        'train_om': float('-inf')
    }
    
    # --- 5. 訓練迴圈 ---
    for epoch in range(cfg.EPOCHS):
        start_time = time.time()
        
        # === Training Phase ===
        model.train()
        train_log = {'loss': 0, 'dice': 0, 'om': 0, 'acc': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for batch in pbar:
            # 搬運資料
            x_prev = batch['prev'].to(device)
            x_curr = batch['curr'].to(device)
            x_next = batch['next'].to(device) if 'next' in batch else None
            y_true = batch['label'].to(device)

            optimizer.zero_grad()
            
            # Forward (Mixed Precision)
            with torch.cuda.amp.autocast():
                logits = model(x_prev, x_curr, x_next)
                loss = criterion(y_true, torch.sigmoid(logits)) # Loss 內部通常預期 0~1 的 input
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 計算指標
            metrics = calculate_metrics(logits, y_true)
            
            # 更新 Log
            train_log['loss'] += loss.item()
            train_log['dice'] += metrics['dice']
            train_log['om'] += metrics['om']
            train_log['acc'] += metrics['acc']
            
            pbar.set_postfix({'loss': loss.item(), 'om': metrics['om']})

        # 平均化 Train Metrics
        for k in train_log: train_log[k] /= len(train_loader)

        # === Validation Phase ===
        model.eval()
        val_log = {'loss': 0, 'dice': 0, 'om': 0, 'acc': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                x_prev = batch['prev'].to(device)
                x_curr = batch['curr'].to(device)
                x_next = batch['next'].to(device) if 'next' in batch else None
                y_true = batch['label'].to(device)

                logits = model(x_prev, x_curr, x_next)
                loss = criterion(y_true, torch.sigmoid(logits))
                
                metrics = calculate_metrics(logits, y_true)
                
                val_log['loss'] += loss.item()
                val_log['dice'] += metrics['dice']
                val_log['om'] += metrics['om']
                val_log['acc'] += metrics['acc']
        
        # 平均化 Val Metrics
        for k in val_log: val_log[k] /= len(val_loader)

        # === 結算與存檔 (還原學長的 4 個 Checkpoint) ===
        epoch_time = time.time() - start_time
        print(f"\nSummary Ep {epoch+1} | Time: {epoch_time:.1f}s")
        print(f"Train | Loss: {train_log['loss']:.4f} | OM: {train_log['om']:.4f} | Dice: {train_log['dice']:.4f}")
        print(f"Val   | Loss: {val_log['loss']:.4f} | OM: {val_log['om']:.4f} | Dice: {val_log['dice']:.4f}")

        # 1. Save Best Val Loss (model_loss.h5)
        if val_log['loss'] < best_metrics['val_loss']:
            best_metrics['val_loss'] = val_log['loss']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_val_loss.pth'))
            print("✅ Saved Best Val Loss Model")

        # 2. Save Best Val OM (model_om.h5)
        if val_log['om'] > best_metrics['val_om']:
            best_metrics['val_om'] = val_log['om']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_val_om.pth'))
            print("✅ Saved Best Val OM Model")

        # 3. Save Best Train Loss (model_train_loss.h5) - 選用
        if train_log['loss'] < best_metrics['train_loss']:
            best_metrics['train_loss'] = train_log['loss']
            # torch.save(model.state_dict(), os.path.join(save_dir, 'best_train_loss.pth'))

        # 4. Save Best Train OM (model_train_om.h5) - 選用
        if train_log['om'] > best_metrics['train_om']:
            best_metrics['train_om'] = train_log['om']
            # torch.save(model.state_dict(), os.path.join(save_dir, 'best_train_om.pth'))

        # 寫入 JSON Log
        log_entry = {
            'epoch': epoch + 1,
            'time': epoch_time,
            'train': train_log,
            'val': val_log
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

if __name__ == '__main__':
    train()