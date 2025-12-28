import os
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader



def save_prediction_image(pred, gt, img, save_path):
    """
    畫出 Input, GT, Pred 的對比圖 (參考學長的 plot_result)
    """
    # 轉回 0~255
    img = (img * 255).astype(np.uint8)
    gt = (gt * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)
    
    # 簡單拼接: [Input, GT, Pred]
    # 你可以依照需求改成疊圖 (Overlay)
    concat = np.hstack([img, gt, pred])
    cv2.imwrite(save_path, concat)

def evaluate_all():
    cfg = Config()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 1. 載入模型
    print("Loading Model...")
    # 這裡你需要先跑一次 train.py 產生 best_model.pth
    model_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}! Please train first.")
        return

    # 計算 Input Channels (需與 Training 一致)
    # 這裡簡單假設你沒開 grad/blur，若是 config 有開要同步邏輯
    dummy_ds = CRNNDataset(cfg.TRAIN_IMG_DIR, n_frames=cfg.N_FRAMES, 
                           use_blur=cfg.USE_BLUR, use_grad=cfg.USE_GRAD)
    in_ch = dummy_ds[0]['curr'].shape[0]

    model_cfg = ModelConfig(
        block_channels=cfg.BLOCK_CHANNELS,
        n_frames=cfg.N_FRAMES,
        input_channels=in_ch,
        num_classes=cfg.NUM_CLASSES
    )
    model = ConcurrentUNet(model_cfg).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. 準備測試資料 (按 Case 處理)
    # 假設 TEST_IMG_DIR 下面是各個 Case 的資料夾
    test_root = './data/all/test'          # 請修改為實際路徑
    test_target = './data/all/test_target' # 請修改為實際路徑
    output_dir = './evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 取得所有 Case 名稱
    case_names = sorted([d for d in os.listdir(test_root) 
                        if os.path.isdir(os.path.join(test_root, d))])
    
    all_results = []
    
    print(f"Found {len(case_names)} cases to evaluate.")

    # 3. 逐一 Case 評估
    for case_name in tqdm(case_names, desc="Evaluating Cases"):
        case_img_dir = os.path.join(test_root, case_name)
        case_target_dir = os.path.join(test_target, case_name) # 假設結構對稱
        
        # 針對單一 Case 建立 Dataset
        # 注意：這裡 root_dir 直接指到 case 資料夾，所以 Dataset 內部的邏輯要能處理
        # 我們之前寫的 Dataset 預設 root_dir 下面是 case folders
        # 為了方便，我們這裡還是指向 root，但在 Dataset 內部我們只會讀取特定 case
        # 或者：簡單一點，我們直接修改 Dataset 讓它接受 list of files
        # 這裡採用最簡單方法：直接用我們寫好的 Dataset，但 root 指向 case 的上一層，
        # 然後用 dataset 內部的 filter? 不，這樣太慢。
        
        # --- 權宜之計：直接在這裡實例化 Dataset 讀取單一資料夾 ---
        # 為了不改動 Dataset.py 太大，建議 Dataset 支援 "Single Case Mode"
        # 只要傳入 case_img_dir 當作 root，且 Dataset 發現下面沒有子資料夾時，就會把它當作單一 case 處理
        # (我之前給你的 Dataset 已經包含了這個邏輯 check: if not case_dirs...)
        
        ds = CRNNDataset(
            root_dir=case_img_dir, 
            target_dir=case_target_dir,
            n_frames=cfg.N_FRAMES,
            use_blur=cfg.USE_BLUR,
            use_grad=cfg.USE_GRAD
        )
        
        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False) # 絕對不能 Shuffle
        
        metrics = VolumeMetrics()
        
        # 建立保存圖片的資料夾
        save_img_dir = os.path.join(output_dir, "plots", case_name)
        os.makedirs(save_img_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                x_prev = batch['prev'].to(device)
                x_curr = batch['curr'].to(device)
                x_next = batch['next'].to(device) if 'next' in batch else None
                y_true = batch['label'].to(device)

                # Inference
                logits = model(x_prev, x_curr, x_next)
                preds = torch.sigmoid(logits)
                preds_bin = (preds > 0.5).float() # 二值化

                # 更新指標
                metrics.update(preds_bin, y_true)

                # 保存圖片 (只存每個 Batch 的第一張示意，避免存太多)
                # 這裡需要小心 tensor 轉 numpy 的維度
                # batch['curr'] 是 (B, C, H, W)，我們只取 Channel 0 (原圖)
                input_img = batch['curr'][0, 0].cpu().numpy() 
                gt_img = y_true[0, 0].cpu().numpy()
                pred_img = preds_bin[0, 0].cpu().numpy()
                
                # 存檔名包含 Batch Index
                save_path = os.path.join(save_img_dir, f"slice_{i*cfg.BATCH_SIZE}.png")
                save_prediction_image(pred_img, gt_img, input_img, save_path)

        # 計算該 Case 的最終分數
        res = metrics.get_metrics()
        res['Case'] = case_name
        all_results.append(res)
        
        # 顯示當前 Case 的 OM
        # tqdm.write(f"Case {case_name}: OM={res['OM']:.4f}")

    # 4. 輸出報表
    df = pd.DataFrame(all_results)
    # 把 Case 移到第一欄
    cols = ['Case'] + [c for c in df.columns if c != 'Case']
    df = df[cols]
    
    csv_path = os.path.join(output_dir, "case_result.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation Complete! Results saved to {csv_path}")
    print(f"Average OM: {df['OM'].mean():.4f}")

if __name__ == '__main__':
    evaluate_all()