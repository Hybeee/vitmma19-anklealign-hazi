import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np

import os
import urllib.request

from config import Args
from vit_pytorch.models.modeling import VisionTransformer, CONFIGS 

def get_model(args: Args, train_labels=None):
    args.logger.info(f"Initializing model: {args.model_name}")

    if args.model_name.lower() == "dummy_baseline":
        if train_labels is None:
            args.logger.error(f"No labels were given for: {args.model_name}")
            return None
        
        model = DummyBaseLine(train_labels=train_labels)
    elif args.model_name.lower() == "anklealign_simple":
        model = AnkleAlignSimple(args)
    elif args.model_name.lower() == "anklealign_medium":
        model = AnkleAlignMedium(args)
    elif args.model_name.lower() == "anklealign_complex":
        model = AnkleAlignComplex(args)
    elif args.model_name.lower() == "anklealign_vit":
        vitargs = args.vitargs
        config = CONFIGS[vitargs.vit_model]
        model = VisionTransformer(
            config=config,
            img_size=args.resolution,
            num_classes=len(args.classes),
            zero_head=True,
            vis=True
        )
        if not os.path.exists(vitargs.pretrained_weights_path):
            args.logger.info(f"Weights not found at: {vitargs.pretrained_weights_path}")

            os.makedirs(os.path.dirname(vitargs.pretrained_weights_path), exist_ok=True)

            try:
                args.logger.info(f"Downloading weights from: {vitargs.download_url}")
                urllib.request.urlretrieve(vitargs.download_url, vitargs.pretrained_weights_path)
                args.logger.info("Download completed successfully.")
            except Exception as e:
                args.logger.error(f"Failed to download weights: {e}")
                return None
        else:
            args.logger.info(f"Weights found at: {vitargs.pretrained_weights_path}")
        model.load_from(np.load(vitargs.pretrained_weights_path))
    else:
        args.logger.error(f"Unknown model name specified in arguments: {args.model_name}")
        return None
    
    if isinstance(model, torch.nn.Module):
        input_size = (args.batch_size, 3, args.resolution, args.resolution)

        model_summary = summary(
            model=model,
            input_size=input_size,
            verbose=0
        )

        args.logger.info(f"Model Architecture Summary:\n\n{model_summary}")
    else:
        args.logger.info("Model is not a standard torch.nn.module. No information available of trainable parameters.")
    
    return model

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

def load_trained_model_from_path(model_path):
    model = torch.load(model_path)
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
    
class AnkleAlignMedium(nn.Module):
    def __init__(self, args: Args):
        super(AnkleAlignMedium, self).__init__()

        num_classes = len(args.classes)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=RuntimeError),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out
    
class AnkleAlignComplex(nn.Module):
    def __init__(self, args: Args):
        super(AnkleAlignComplex, self).__init__()

        num_classes = len(args.classes)
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x