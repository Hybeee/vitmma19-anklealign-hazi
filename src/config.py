import torch
import os
import logging

class Args:
    def __init__(self):
        self.epochs = 100
        self.batch_size = 32
        self.lr = 1e-4
        self.early_stopping = 10

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.output_dir = "tbd"
        self.model_dir = "tbd"
        self.data_dir = "tbd"

        self.f_loss = torch.nn.CrossEntropyLoss
        self.optimizer = "adam"