from torch.utils.data import Dataset
import numpy as np
import cv2
import torch


def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # H. W
    xray = xray.astype(np.float32) / 255
    xray = xray.reshape((1, *xray.shape))
    return xray


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    mask = mask.reshape((1, *mask.shape))  # returns (1, H, W)
    return mask


class KneeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = read_xray(self.df['xray_path'].iloc[index])
        mask = read_mask(self.df['mask_path'].iloc[index])

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        res = {
            'image': image,
            'mask': mask
        }
        return res


class KneeDatasetTest(Dataset):
    """Dataset for test/inference without masks."""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = read_xray(self.df['xray_path'].iloc[index])
        return {'image': image}

