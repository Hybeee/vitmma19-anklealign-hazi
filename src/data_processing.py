import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import Args

class AnkleAlignDataset(Dataset):
    def __init__(self, images, labels, transform):
        super(AnkleAlignDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return self.transform(image), torch.tensor(label, dtype=torch.long)
    
def get_loader(args: Args, split: str,
               images, labels):
    if split == "train":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resolution, args.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=args.cj_brightness, contrast=args.cj_contrast, saturation=args.cj_saturation),
            transforms.RandomRotation(args.rotation),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
        ])

        train_dataset = AnkleAlignDataset(images=images, labels=labels,
                                          transform=train_transform)
        
        train_loader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, drop_last=True)
        
        return train_loader
    
    else: # split == val/train
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
        ])

        dataset = AnkleAlignDataset(images=images, labels=labels,
                                    transform=transform)
        
        data_loader = DataLoader(dataset, args.batch_size,
                                 shuffle=False)
        
        return data_loader