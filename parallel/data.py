import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as T

class SkinCancerDataset(Dataset):
    def __init__(
        self,
        dir: str,
        label: str,
        mapping: dict,
        transform: list,
    ):
        self.transform = T.Compose(transform)
        self.mapping = mapping
        self.data, self.label = [], []
        with open(label, 'r') as fp:
            label_csv = dict([tuple(line.strip().split(',')) for line in fp.readlines()])
            for filename in os.listdir(dir):
                self.data.append(os.path.join(dir, filename))
                self.label.append(label_csv[filename.split('.')[0]])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.data[idx]))
        label = self.label[idx]
        index = torch.tensor(self.mapping[label], dtype=torch.long)
        return image, index, label
