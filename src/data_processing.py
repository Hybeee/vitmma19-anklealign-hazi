import torch
from torch.utils.data import Dataset

class AnkleAlignDataset(Dataset):
    def __init__(self, images, labels, transform):
        super(AnkleAlignDataset, self)
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return self.transform(image), torch.tensor(label, dtype=torch.long)