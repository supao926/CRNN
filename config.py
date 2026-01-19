import os

class Config:
    # --- 路徑設定 ---
    # 請修改為你實際的路徑
    TRAIN_IMG_DIR = r"C:\Users\super\Documents\crnn\data\train\OR"
    TRAIN_MASK_DIR = r"C:\Users\super\Documents\crnn\data\train\GT"
    TEST_IMG_DIR = r"C:\Users\super\Documents\crnn\data\test\OR"
    TEST_MASK_DIR = r"C:\Users\super\Documents\crnn\data\test\GT"
    CHECKPOINT_DIR = r"C:\Users\super\Documents\crnn\data\checkpoint"

    # --- 硬體與訓練設定 ---
    DEVICE = "cuda"      # 'cuda' or 'cpu'
    EPOCHS = 50
    BATCH_SIZE = 8       # 建議至少 4，因為 Loss 會做 Shift 操作
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4      # Windows 下若報錯可改為 0
    
    # --- 模型設定 ---
    N_FRAMES = 3         # 2 或 3
    BLOCK_CHANNELS = [32, 64, 128, 256] # 模型深度
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 1      # 二元分割設為 1
    
    # --- 特徵工程 ---
    USE_BLUR = True
    USE_GRAD = True