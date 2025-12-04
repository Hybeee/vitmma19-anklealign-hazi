import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Args
import utils

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

def create_and_save_splits(args:Args, images, labels):
    save_dir = os.path.join(args.data_dir, "splits")
    os.makedirs(save_dir, exist_ok=True)

    X = images
    Y = labels

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=args.seed,
        stratify=Y
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=0.5,
        random_state=args.seed,
        stratify=Y_temp
    )

    np.save(os.path.join(save_dir, "train_images.npy"), X_train)
    np.save(os.path.join(save_dir, "train_labels.npy"), Y_train)
    np.save(os.path.join(save_dir, "val_images.npy"), X_val)
    np.save(os.path.join(save_dir, "val_labels.npy"), Y_val)
    np.save(os.path.join(save_dir, "test_images.npy"), X_test)
    np.save(os.path.join(save_dir, "test_labels.npy"), Y_test)

    args.logger.info("Created splits")
    args.logger.info(f"Train images and labels shape: {X_train.shape} - {Y_train.shape}")
    args.logger.info(f"Val images and labels shape: {X_val.shape} - {Y_val.shape}")
    args.logger.info(f"Test images and labels shape: {X_test.shape} - {Y_test.shape}")

    if args.save_split_results:
        output_plots_dir = os.path.join(args.output_dir, "plots")
        utils.save_split_results(args, Y_train, Y_val, Y_test,
                                 output_plots_dir)
        args.logger.info("Saved split result plots")

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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp

    args = Args()
    args.output_dir = os.path.join("outputs", f"{timestamp}_{args.model_alias}")
    output_plots_dir = os.path.join(args.output_dir, "plots")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    log_dir = args.output_dir
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=log_dir)

    images = np.load(os.path.join(args.data_dir, "cleaned_numpy_data", "images.npy"))
    labels = np.load(os.path.join(args.data_dir, "cleaned_numpy_data", "labels.npy"))

    create_and_save_splits(args=args,
                           images=images,
                           labels=labels)

if __name__ == "__main__":
    main()