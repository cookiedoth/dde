import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from gutils.basic import torch_randint, to_tuple2


class SatMapDataset(Dataset):
    def __init__(self, path, image_size, flip_cond=False):
        self.sat_data = np.load(path)['sat']
        self.map_data = np.load(path)['map']
        if flip_cond:
            self.sat_data, self.map_data = self.map_data, self.sat_data
        self.shape = (self.map_data.shape[2], self.map_data.shape[3])
        self.image_size = to_tuple2(image_size)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.sat_data)

    def __getitem__(self, idx):
        H, W = self.image_size
        h0 = torch_randint(0, self.shape[0] - H + 1)
        w0 = torch_randint(0, self.shape[1] - W + 1)
        sat = self.transform(self.sat_data[idx].transpose(1, 2, 0))
        mp = self.transform(self.map_data[idx].transpose(1, 2, 0))
        return sat[:, h0:h0+H, w0:w0+W], mp[:, h0:h0+H, w0:w0+W]


def merge_sat_map(dataset, indices, device):
    sats = []
    maps = []
    for i in indices:
        s, m = dataset[i]
        sats.append(s)
        maps.append(m)
    return torch.stack(sats, dim=0).to(device), \
           torch.stack(maps, dim=0).to(device)

