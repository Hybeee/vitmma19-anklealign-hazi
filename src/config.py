import torch
import os
import logging

MODEL_OPTIONS = {
    "baseline": "dummy_baseline",
    "simple": "anklealign_simple",
    "complex": "anklealign_complex"
}

class Args:
    def __init__(self):
        self.seed = 42

        self.model_choice = "complex"
        self.model_name = MODEL_OPTIONS[self.model_choice]
        self.model_alias = self._get_name_alias()

        self.classes = {
            0: 'Pronacio',
            1: 'Neutralis',
            2: 'Szupinacio'
        }

        self.reverse_classes = {
            'Pronacio': 0,
            'Neutralis': 1,
            'Szupinacio': 2
        }

        self.similarity_threshold = 1e-4
        self.save_dupes = True
        self.sharpness_threshold = 40
        self.min_contrast_std = 10.0
        self.min_brightness_mean = 40.0
        self.save_lq = True

        self.save_split_results = True
        self.resolution = 224
        self.cj_brightness = 0.2
        self.cj_contrast = 0.2
        self.cj_saturation = 0.2
        self.rotation = 15
        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        self.epochs = 100
        self.batch_size = 16
        self.lr = 1e-5
        self.early_stopping = 10

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loss_name = "ce"
        self.use_label_weights = True
        self.optimizer = "adam"

        self.output_dir = "outputs"
        self.data_dir = "data"

        self.logger = None

    def _get_name_alias(self):
        if self.model_name.lower() == "dummy_baseline":
            return "ds"
        elif self.model_name.lower() == "anklealign_simple":
            return "aa_s"
        elif self.model_name.lower() == "anklealign_complex":
            return "aa_c"
        else:
            return "na"