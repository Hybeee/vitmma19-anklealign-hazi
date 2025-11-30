import os
import logging

import matplotlib.pyplot as plt
import numpy as np

from config import Args

def save_plots(args: Args, train_loss, train_accuracy, val_loss, val_accuracy):
    plt.figure()
    plt.plot(train_loss, label='Train Loss', color='Green')
    plt.plot(val_loss, label='Validation Loss', color='Blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "plots", 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(train_accuracy, label='Train Accuracy', color='Green')
    plt.plot(val_accuracy, label='Validation Accuracy', color='Blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "plots", 'accuracy_curve.png'))
    plt.close()

def get_logger(timestamp: str, log_dir: str="logs"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(timestamp)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, f"{timestamp}.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def plot_label_distribution(args: Args, class_counts, output_plots_dir):
    labels = sorted([k for k in class_counts.keys()])
    counts = [class_counts[label] for label in labels]

    total = sum(counts)

    x_labels = [f'{args.classes[label]}' for label in labels]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x_labels, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.title("AnkleAlign Dataset Class Distribution")
    plt.xlabel("Ankle Alignment Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + (total * 0.01),
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_dir, "label_distribution.png"))
    plt.close()

def plot_aspect_ratio(aspect_ratios, output_plots_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(aspect_ratios, bins=25, color='#8c564b', edgecolor='black', alpha=0.7)
    mean_ratio = np.mean(aspect_ratios)
    plt.axvline(mean_ratio, color='r', linestyle='dashed', linewidth=1, label=f'Mean Ratio: {mean_ratio:.2f}')

    plt.title("Distribution of Original Image Aspect Ratios")
    plt.xlabel("Aspect Ration (Height/Width)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_dir, "aspect_ratio_distribution.png"))
    plt.close()

def load_model(model_path: str):
    # TODO
    pass