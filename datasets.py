import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import numpy as np
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler


# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data") -> None:
#         super().__init__()
        
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
        
#         self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, i):
#         if hasattr(self, "y"):
#             return self.X[i], self.y[i], self.subject_idxs[i]
#         else:
#             return self.X[i], self.subject_idxs[i]
        
#     @property
#     def num_channels(self) -> int:
#         return self.X.shape[1]
    
#     @property
#     def seq_len(self) -> int:
#         return self.X.shape[2]
    

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200, normalize: bool = True, denoise: bool = True) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self.resample_data(self.X, resample_rate)
        
        if denoise:
            # フィルタリング
            self.X = self.denoise_data(self.X, fs=resample_rate)
        
        if normalize:
            # スケーリングとベースライン補正
            self.X = self.normalize_and_baseline_correct(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
    def resample_data(self, X, new_rate):
        # リサンプリング処理
        num_samples = X.shape[2]
        new_num_samples = int(num_samples * new_rate / 1000)
        resampled_X = signal.resample(X, new_num_samples, axis=2)
        return torch.tensor(resampled_X, dtype=torch.float32)
    
    def denoise_data(self, X, lowcut=0.5, highcut=40.0, fs=200.0, order=5):
        # フィルタリング処理
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        X_denoised = signal.lfilter(b, a, X.cpu().numpy(), axis=-1)
        return torch.tensor(X_denoised, dtype=torch.float32)
    
    def normalize_and_baseline_correct(self, X):
        # スケーリングとベースライン補正処理
        X_np = X.cpu().numpy()
        num_trials, num_channels, num_samples = X_np.shape
        
        # スケーリング
        scaler = StandardScaler()
        X_scaled = np.zeros_like(X_np)
        for i in range(num_channels):
            X_scaled[:, i, :] = scaler.fit_transform(X_np[:, i, :].reshape(-1, 1)).reshape(num_trials, num_samples)
        
        # ベースライン補正
        baseline = X_scaled[:, :, :100].mean(axis=2, keepdims=True)
        X_corrected = X_scaled - baseline
        
        return torch.tensor(X_corrected, dtype=torch.float32)