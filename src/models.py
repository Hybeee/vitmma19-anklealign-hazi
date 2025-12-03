import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os

from config import Args

def get_model(args: Args, train_labels=None):
    args.logger.info(f"Initializing model: {args.model_name}")

    if args.model_name.lower() == "dummy_baseline":
        if train_labels is None:
            args.logger.error(f"No labels were given for: {args.model_name}")
            return None
        
        return DummyBaseLine(train_labels=train_labels)
    elif args.model_name.lower() == "anklealign_simple":
        return AnkleAlignSimple(args)
    else:
        args.logger.error(f"Unknown model name specified in arguments: {args.model_name}")
        return None

def load_trained_model(args: Args, train_labels=None):
    args.logger.info(f"Loading trained model: {args.model_name}")

    if args.model_name.lower() == "dummy_baseline":
        if train_labels is None:
            args.logger.error(f"No labels were given for: {args.model_name}")
            return None
        return DummyBaseLine(train_labels=train_labels)
    else:
        model = torch.load(os.path.join(args.output_dir, "model.pth"))
        return model

class DummyBaseLine(nn.Module):
    def __init__(self, train_labels):
        super(DummyBaseLine, self).__init__()

        self.classes = np.unique(train_labels)
        self.num_classes = len(self.classes)
        self.majority_class = self._calculate_majority_class(train_labels)

        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def _calculate_majority_class(self, train_labels):
        unique, counts = np.unique(train_labels, return_counts=True)

        majority_index = np.argmax(counts)
        return unique[majority_index]
    
    def forward(self, x):
        batch_size = x.size(0)

        logits = torch.full((batch_size, self.num_classes), -100.0, device=self.dummy_param.device)

        logits[:, self.majority_class] = 100.0

        return logits
    
class AnkleAlignSimple(nn.Module):
    def __init__(self, args: Args):
        super(AnkleAlignSimple, self).__init__()

        num_classes = len(args.classes)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x