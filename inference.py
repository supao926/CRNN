import os
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from dataset import CRNNDataset
from model import ConcurrentUNet
from metrics import VolumeMetrics

# --- 視覺化工具 (保持不變) ---
def normalize_for_display(img):
    img = img.astype(np.float32)
    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)

def save_prediction_image(pred, gt, img, save_path):
    img_show = normalize_for_display(img)
    gt_show = (gt * 255).astype(np.uint8)
    pred_show = (pred * 255).astype(np.uint8)
    concat = np.hstack([img_show, gt_show, pred_show])
    cv2.imwrite(save_path, concat)

def evaluate_all():
    cfg = Config()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 強制設定通道數 (因應你的 Config 設定)
    cfg.input_channels = 6 
    print(f"Force set input channels to: {cfg.input_channels}")

    print(f"Target Image Dir: {cfg.TEST_IMG_DIR}")
    if not os.path.exists(cfg.TEST_IMG_DIR):
        print(f"Error: Test dir not found: {cfg.TEST_IMG_DIR}")
        return

    print("Loading Model...")
    model_path = os.path.join(cfg.CHECKPOINT_DIR, "best_val_om.pth") 
    if not os.path.exists(model_path):
        print(f"Checkpoint not found at {model_path}, trying best_val_loss.pth")
        model_path = os.path.join(cfg.CHECKPOINT_DIR, "best_val_loss.pth")
    
    model = ConcurrentUNet(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    output_dir = './evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    case_names = sorted([d for d in os.listdir(cfg.TEST_IMG_DIR) 
                         if os.path.isdir(os.path.join(cfg.TEST_IMG_DIR, d))])
    
    all_results = []
    print(f"Found {len(case_names)} cases. Starting inference...")

    for case_name in tqdm(case_names, desc="Evaluating"):
        case_img_dir = os.path.join(cfg.TEST_IMG_DIR, case_name)
        case_target_dir = os.path.join(cfg.TEST_MASK_DIR, case_name)
        
        try:
            ds = CRNNDataset(
                root_dir=case_img_dir, 
                target_dir=case_target_dir,
                n_frames=cfg.N_FRAMES,
                use_blur=cfg.USE_BLUR,
                use_grad=cfg.USE_GRAD
            )
        except Exception as e:
            print(f"Skipping {case_name}: {e}")
            continue
        
        if len(ds) == 0: continue

        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        metrics = VolumeMetrics()
        
        save_img_dir = os.path.join(output_dir, "plots", case_name)
        os.makedirs(save_img_dir, exist_ok=True)

        with torch.no_grad():
            # i 是 batch 的編號
            for i, batch in enumerate(loader):
                x_prev = batch['prev'].to(device)
                x_curr = batch['curr'].to(device)
                x_next = batch['next'].to(device) if 'next' in batch else None
                y_true = batch['label'].to(device)

                logits = model(x_prev, x_curr, x_next)
                preds = torch.sigmoid(logits)
                preds_bin = (preds > 0.5).float()

                metrics.update(preds_bin, y_true)

                # === 修改部分：遍歷 Batch 中的每一張圖 ===
                # 取得當前 Batch 的大小 (最後一個 Batch 可能小於 cfg.BATCH_SIZE)
                current_batch_size = x_curr.size(0)

                for j in range(current_batch_size):
                    # 取出單張切片 (移除 Batch 維度)
                    input_img = batch['curr'][j, 0].cpu().numpy() 
                    gt_img = y_true[j, 0].cpu().numpy()
                    pred_img = preds_bin[j, 0].cpu().numpy()
                    
                    # 計算全域切片編號，確保檔名連續不重複
                    global_slice_idx = i * cfg.BATCH_SIZE + j
                    
                    save_path = os.path.join(save_img_dir, f"slice_{global_slice_idx:04d}.png")
                    save_prediction_image(pred_img, gt_img, input_img, save_path)
                # ======================================

        res = metrics.get_metrics()
        res['Case'] = case_name
        all_results.append(res)
        
        tqdm.write(f"Case: {case_name} | Dice: {res.get('Dice', 0):.4f} | OM: {res.get('OM', 0):.4f}")

    df = pd.DataFrame(all_results)
    if not df.empty:
        cols = ['Case'] + [c for c in df.columns if c != 'Case']
        df = df[cols]
        csv_path = os.path.join(output_dir, "case_result.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        if 'OM' in df.columns:
            print(f"Average OM: {df['OM'].mean():.4f}")

if __name__ == '__main__':
    evaluate_all()