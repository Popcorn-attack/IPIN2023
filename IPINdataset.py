import torch
from torch.utils.data import DataLoader,Dataset
import re
import numpy as np


    


class IPINDataset(Dataset):
    def __init__(self, data_path, label_path):

        self.sensorslist =  data_path
        self.sensorslabel = label_path

    def __getitem__(self, idx):
        raw_data = self.sensorslist[idx]  
        label = self.sensorslabel[idx]
        raw_data = torch.tensor(raw_data,dtype=float)
        label = torch.tensor(label,dtype=float)
        return raw_data, label

    def __len__(self):
        return len(self.sensorslist)
