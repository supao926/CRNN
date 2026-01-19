import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

# Correct Imports
from config import Config
from dataset import CRNNDataset
from model import ConcurrentUNet
from loss import WeightedConsecutiveLoss
from metrics import calculate_metrics

def train():
    cfg = Config()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    log_file = os.path.join(cfg.CHECKPOINT_DIR, 'training_log.json')
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Start Training on {device}...")

    # Data
    full_ds = CRNNDataset(
        root_dir=cfg.TRAIN_IMG_DIR,
        target_dir=cfg.TRAIN_MASK_DIR,
        n_frames=cfg.N_FRAMES,
        use_blur=cfg.USE_BLUR,
        use_grad=cfg.USE_GRAD
    )
    
    # Check if data loaded
    if len(full_ds) == 0:
        print("Error: No data found. Check TRAIN_IMG_DIR.")
        return

    # Dynamic Input Channel Detection
    sample_ch = full_ds[0]['curr'].shape[0]
    # Inject input_channels into config for Model
    cfg.input_channels = sample_ch 
    print(f"Detected Input Channels: {sample_ch}")
    
    val_size = int(len(full_ds) * 0.2)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    # drop_last=True is crucial for Consecutive Loss to avoid shape mismatch on last small batch
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False)

    # Model
    model = ConcurrentUNet(cfg).to(device)
    criterion = WeightedConsecutiveLoss(loss_ratio=0.5, consecutive=cfg.N_FRAMES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') # Update for newer PyTorch

    best_metrics = {'val_loss': float('inf'), 'val_om': float('-inf')}
    
    for epoch in range(cfg.EPOCHS):
        start_time = time.time()
        
        # === Training ===
        model.train()
        train_log = {'loss': 0, 'dice': 0, 'om': 0, 'acc': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for batch in pbar:
            x_prev = batch['prev'].to(device)
            x_curr = batch['curr'].to(device)
            x_next = batch['next'].to(device) if 'next' in batch else None
            y_true = batch['label'].to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(x_prev, x_curr, x_next)
                # CRITICAL: Loss needs Probabilities, Metrics need Logits (internally handled)
                probs = torch.sigmoid(logits)
                loss = criterion(y_true, probs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate metrics using Logits
            metrics = calculate_metrics(logits, y_true)
            
            train_log['loss'] += loss.item()
            train_log['dice'] += metrics['dice']
            train_log['om'] += metrics['om']
            train_log['acc'] += metrics['acc']
            pbar.set_postfix({'loss': loss.item(), 'om': metrics['om']})

        for k in train_log: train_log[k] /= len(train_loader)

        # === Validation ===
        model.eval()
        val_log = {'loss': 0, 'dice': 0, 'om': 0, 'acc': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                x_prev = batch['prev'].to(device)
                x_curr = batch['curr'].to(device)
                x_next = batch['next'].to(device) if 'next' in batch else None
                y_true = batch['label'].to(device)

                logits = model(x_prev, x_curr, x_next)
                probs = torch.sigmoid(logits)
                loss = criterion(y_true, probs)
                metrics = calculate_metrics(logits, y_true)
                
                val_log['loss'] += loss.item()
                val_log['dice'] += metrics['dice']
                val_log['om'] += metrics['om']
                val_log['acc'] += metrics['acc']
        
        for k in val_log: val_log[k] /= len(val_loader)

        # Save Logic
        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_log['loss']:.4f} OM: {train_log['om']:.4f}")
        print(f"Val Loss: {val_log['loss']:.4f} OM: {val_log['om']:.4f}")

        if val_log['loss'] < best_metrics['val_loss']:
            best_metrics['val_loss'] = val_log['loss']
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, 'best_val_loss.pth'))
            print("Saved Best Loss Model")

        if val_log['om'] > best_metrics['val_om']:
            best_metrics['val_om'] = val_log['om']
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, 'best_val_om.pth'))
            print("Saved Best OM Model")
            
        # Log to file
        with open(log_file, 'a') as f:
            f.write(json.dumps({'epoch': epoch+1, 'train': train_log, 'val': val_log}) + "\n")

if __name__ == '__main__':
    train()