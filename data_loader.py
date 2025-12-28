import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class CRNNDataset(Dataset):
    def __init__(self, root_dir, target_dir=None, n_frames=3, is_train=True, 
                 use_blur=False, use_grad=False, use_hist=False):
        """
        Args:
            root_dir: 訓練圖片的根目錄 (裡面包含各個 Case 的資料夾，或直接是圖片)
            target_dir: Ground Truth 的根目錄
            n_frames: 2 或 3 (對應學長的 consecutive)
            use_blur: 是否使用高斯模糊特徵 (學長的 args.blur)
            use_grad: 是否使用梯度特徵 (學長的 args.gradient_preprocess)
            use_hist: 是否使用直方圖特徵 (學長的 input_hist)
        """
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.n_frames = n_frames
        self.is_train = is_train
        
        # 特徵工程開關
        self.use_blur = use_blur
        self.use_grad = use_grad
        self.use_hist = use_hist

        # 1. 建立滑動視窗索引 (Sliding Window Indexing)
        # 格式: [ (prev_path, curr_path, next_path), label_path ]
        self.samples = self._make_dataset()

    def _make_dataset(self):
        """
        模擬學長的 create_crnn_gen_file_list
        """
        samples = []
        
        # 假設 root_dir 裡面是各個 Case 的資料夾 (例如: case1, case2...)
        # 如果你的結構是直接一堆 png，請把 root_dir 當作單一 case 處理
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Directory not found: {self.root_dir}")

        # 這裡假設 root_dir 下面有很多子資料夾，每個資料夾代表一個 Volume
        # 如果你的資料結構是扁平的 (直接在 root_dir 下就是 1.png, 2.png...)
        # 那就把 case_dirs 設為 [self.root_dir]
        case_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        """
        以上的寫法等價於
        case_dirs = []
        for d in os.listdir(self.root_dir):
        full_path = os.path.join(self.root_dir, d)
        if os.path.isdir(full_path):
        case_dirs.append(full_path)
        """
        # 如果找不到子資料夾，就假設 root_dir 本身就是一個 case
        ## 視為單一case 
        if not case_dirs:
            case_dirs = [self.root_dir]

        for case_path in case_dirs:
            # 讀取該 Case 下所有 png 並排序 (學長的 create_img_path_list 邏輯)
            # 排序很重要！必須是 1.png, 2.png, 3.png...
            imgs = sorted([f for f in os.listdir(case_path) if f.endswith('.png')],
                          key=lambda x: int(os.path.splitext(x)[0]))
            
            full_img_paths = [os.path.join(case_path, f) for f in imgs]
            ## 此時full_img_paths為儲存此迴圈中的所有path之排序
            
            if self.target_dir:
                # 假設 Target 的結構跟 Input 一模一樣
                case_name = os.path.basename(case_path)
                ## case_path是在root_dir後 png檔案外面那層資料夾
                target_case_path = os.path.join(self.target_dir, case_name)
                # 如果 target 結構是扁平的，要自己調整這裡
                if not os.path.exists(target_case_path): 
                     target_case_path = self.target_dir # Fallback
                
                full_target_paths = [os.path.join(target_case_path, f) for f in imgs]
            
            # 製作滑動視窗
            # 如果 n=3, 會有 [i, i+1, i+2]
            for i in range(len(full_img_paths) - self.n_frames + 1):
                window_imgs = full_img_paths[i : i + self.n_frames]
                
                # Label 對應的是中間那張 (curr) 或是最後一張?
                # 學長邏輯: return batch_Y.append(tmp_y[1]) -> n=3 時取中間
                # n=2 時取後面那個? 學長代碼: yield ... np.array(batch_Y) (batch_Y append tmp_y[1])
                # 結論：Label 總是取 window 中的 index=1 (第二張)
                
                label_path = None
                if self.target_dir:
                    # 取 window 中間 (index 1) 的 label
                    # 如果 n=2, index 1 就是第二張 (Curr)
                    label_idx = i + 1 
                    label_path = full_target_paths[label_idx]
                
                samples.append((window_imgs, label_path))
                
        return samples ## return的samples 會是一個系列的切片位置 對應這組系列切片應該產出的GT位置

    def _load_image(self, path): ## 讀image並且做標準化
        # 替換 scipy.misc.imread -> cv2.imread
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = img.astype(np.float32)
        
        # Normalization (0~1)
        # 學長邏輯: (img - min) / (max - min)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = img * 0 # 避免除以 0
            
        return img

    def _preprocess(self, img_np):
        """
        重現學長的 kw_preprocessing (Blur, Gradient)
        img_np shape: (H, W)
        Return shape: (C, H, W) -> C 可能是 1, 3, 5... 取決於特徵
        """
        processed_channels = [img_np] # 原始圖
        
        # 1. Blur (高斯模糊)
        if self.use_blur:
            # 學長用的是 sigma=[2], kernel=7
            blur = cv2.GaussianBlur(img_np, (7, 7), 2)
            # Normalize blur
            b_min, b_max = blur.min(), blur.max()
            if b_max > b_min:
                blur = (blur - b_min) / (b_max - b_min)
            processed_channels.append(blur)
            
        # Stack 起來變成 (H, W, C_temp) 為了做 Gradient
        feature_stack = np.stack(processed_channels, axis=-1) 
        
        # 2. Gradient (Sobel + Laplacian)
        if self.use_grad:
            grad_channels = []
            ksize = 5
            # 對每個現有的 channel (Original, Blur) 做 gradient
            for i in range(feature_stack.shape[-1]):
                src = feature_stack[:, :, i]
                
                # Laplacian
                lap = cv2.Laplacian(src, cv2.CV_32F, ksize=ksize)
                # Normalize Lap
                l_min, l_max = lap.min(), lap.max()
                if l_max > l_min: lap = (lap - l_min) / (l_max - l_min)
                
                # Sobel
                sobelx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize)
                sobel = np.maximum(np.abs(sobelx), np.abs(sobely))
                # Normalize Sobel
                s_min, s_max = sobel.min(), sobel.max()
                if s_max > s_min: sobel = (sobel - s_min) / (s_max - s_min)
                
                grad_channels.append(sobel)
                grad_channels.append(lap)
            
            # 將 Gradient 特徵加回去
            # 學長邏輯：[Original, Blur, Sobel_Org, Lap_Org, Sobel_Blur, Lap_Blur]
            # 這裡我們簡化處理，全部疊加
            processed_channels.extend(grad_channels)

        # Final Stack: (H, W, C) -> Transpose to (C, H, W) for PyTorch
        final_img = np.stack(processed_channels, axis=0) 
        return torch.from_numpy(final_img).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_paths, label_path = self.samples[idx]
        
        # 讀取並處理 Input (Prev, Curr, Next)
        tensors = []
        for p in window_paths:
            img = self._load_image(p)    # Load & Norm
            tensor = self._preprocess(img) # Blur & Grad & ToTensor
            tensors.append(tensor) ## tensor w/ the shape of (C,H,W)
            
        # 根據 n_frames 組裝 dict
        sample = {}
        if self.n_frames == 3:
            sample = {
                'prev': tensors[0],
                'curr': tensors[1],
                'next': tensors[2]
            }
        elif self.n_frames == 2:
            sample = {
                'prev': tensors[0],
                'curr': tensors[1]
            }

        # 讀取 Label
        if self.target_dir and label_path:
            # Label 不需要做 Blur/Grad，只需要讀取跟 Normalize (通常是 0/1)
            lbl = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            lbl = (lbl / 255.0).astype(np.float32) # 轉成 0~1
            lbl = torch.from_numpy(lbl).unsqueeze(0) # (1, H, W)
            sample['label'] = lbl
            
        return sample
    

