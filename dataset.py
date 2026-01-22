import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CRNNDataset(Dataset):
    def __init__(self, root_dir, target_dir=None, n_frames=3, is_train=True, 
                 use_blur=False, use_grad=False, use_hist=False):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.n_frames = n_frames
        self.is_train = is_train
        self.use_blur = use_blur
        self.use_grad = use_grad
        self.use_hist = use_hist
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Directory not found: {self.root_dir}")

        # 判斷 root_dir 是包含多個 Case 還是單一 Case
        # 邏輯：如果 root_dir 下面有資料夾，視為多 Case；否則視為單一 Case
        subdirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) 
                   if os.path.isdir(os.path.join(self.root_dir, d))]
        
        case_dirs = subdirs if subdirs else [self.root_dir]

        for case_path in case_dirs:
            # 讀取並排序圖片
            imgs = sorted([f for f in os.listdir(case_path) if f.endswith('.png')],
                          key=lambda x: int(os.path.splitext(x)[0]))
            ## key = lambda x : int(os.path.splitext(x)[0]) key為sorted看得值 lambda x : int將檔名視為 x轉化成int 並且由.來分割前後

            full_img_paths = [os.path.join(case_path, f) for f in imgs]
            
            # 處理 Label 路徑
            full_target_paths = []
            if self.target_dir:
                case_name = os.path.basename(case_path)
                # 嘗試尋找對應的 Label 資料夾
                target_case_path = os.path.join(self.target_dir, case_name)
                # Fallback: 如果找不到對應資料夾，假設 target_dir 本身就是該 case 的 label 目錄
                if not os.path.exists(target_case_path):
                    target_case_path = self.target_dir 
                
                full_target_paths = [os.path.join(target_case_path, f) for f in imgs]
            
            # 製作滑動視窗
            if len(full_img_paths) >= self.n_frames:
                for i in range(len(full_img_paths) - self.n_frames + 1):
                    window_imgs = full_img_paths[i : i + self.n_frames]
                    
                    label_path = None
                    if self.target_dir:
                        # Label 取視窗中間 (n=3 -> idx 1, n=2 -> idx 1)
                        label_idx = i + 1
                        if label_idx < len(full_target_paths):
                            label_path = full_target_paths[label_idx]
                        else:
                        # --- 新增這段詳細警告 ---
                            curr_img_name = os.path.basename(window_imgs[1])
                            print(f"[Missing GT] Input: {curr_img_name} (in {os.path.basename(case_path)}) has no matching label!")
                        
                            label_path = None
                    samples.append((window_imgs, label_path))
                
        return samples

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = img.astype(np.float32)
        
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = img * 0
        return img

    def _preprocess(self, img_np):
        processed_channels = [img_np]
        
        if self.use_blur:
            blur = cv2.GaussianBlur(img_np, (7, 7), 2)
            b_min, b_max = blur.min(), blur.max()
            if b_max > b_min: blur = (blur - b_min) / (b_max - b_min)
            processed_channels.append(blur)
            
        feature_stack = np.stack(processed_channels, axis=-1)
        
        if self.use_grad:
            grad_channels = []
            ksize = 5
            for i in range(feature_stack.shape[-1]):
                src = feature_stack[:, :, i]
                lap = cv2.Laplacian(src, cv2.CV_32F, ksize=ksize)
                l_min, l_max = lap.min(), lap.max()
                if l_max > l_min: lap = (lap - l_min) / (l_max - l_min)
                
                #cv.Sobel(	src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]	) ->	dst
                sobelx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize)
                sobel = np.maximum(np.abs(sobelx), np.abs(sobely))
                s_min, s_max = sobel.min(), sobel.max()
                if s_max > s_min: sobel = (sobel - s_min) / (s_max - s_min)
                
                grad_channels.append(sobel)
                grad_channels.append(lap)
            processed_channels.extend(grad_channels)

        final_img = np.stack(processed_channels, axis=0) 
        return torch.from_numpy(final_img).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_paths, label_path = self.samples[idx]
        tensors = []
        for p in window_paths:
            img = self._load_image(p)
            tensor = self._preprocess(img)
            tensors.append(tensor)
            
        sample = {}
        if self.n_frames == 3:
            sample = {'prev': tensors[0], 'curr': tensors[1], 'next': tensors[2]}
        elif self.n_frames == 2:
            sample = {'prev': tensors[0], 'curr': tensors[1]}

        if self.target_dir and label_path:
            lbl = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if lbl is not None:
                lbl = (lbl / 255.0).astype(np.float32)
                lbl = torch.from_numpy(lbl).unsqueeze(0)
                sample['label'] = lbl
            else:
                # 預防讀不到 label 的情況
                sample['label'] = torch.zeros((1, tensors[0].shape[1], tensors[0].shape[2]))
            
        return sample