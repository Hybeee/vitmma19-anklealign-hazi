import torch
import os
import logging

class Args:
    def __init__(self):
        self.classes = [] # TODO: fill up

        self.epochs = 100
        self.batch_size = 32
        self.lr = 1e-4
        self.early_stopping = 10

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.f_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = "adam"

        self.resolution = 224
        self.cj_brightness = 0.2
        self.cj_contrast = 0.2
        self.cj_saturation = 0.2
        self.rotation = 15
        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        self.output_dir = "tbd"
        self.model_dir = "tbd"
        self.data_dir = "tbd"

        self.logger = None