import os
import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

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

        fh = logging.FileHandler(os.path.join(log_dir, f"run.log"))
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

def save_dupes(images, i, j, output_plots_dir):
    dupes_plot_dir = os.path.join(output_plots_dir, "dupes")
    os.makedirs(dupes_plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(images[i])
    axes[0].set_title(f"Original (Index {i})")
    axes[0].axis('off')
    
    axes[1].imshow(images[j])
    axes[1].set_title(f"Duplicate (Index {j}) - EXCLUDED")
    axes[1].axis('off')
    
    fig.suptitle(f"Duplicate Pair: Index {i} vs Index {j}", fontsize=16)

    plot_filename = f"dupe_{i}_vs_{j}.png"
    plt.savefig(os.path.join(dupes_plot_dir, plot_filename))
    plt.close(fig)

def save_lq(img, idx, reason, output_plots_dir):
    lq_images_dir = os.path.join(output_plots_dir, "lq_images")
    os.makedirs(lq_images_dir, exist_ok=True)

    plt.imshow(img)
    plt.title(reason)
    plt.savefig(os.path.join(lq_images_dir, f"img_{idx}.png"))
    plt.close()

def save_split_results(args: Args, Y_train, Y_val, Y_test,
                       output_plots_dir):
    train_counts = Counter(Y_train)
    val_counts = Counter(Y_val)
    test_counts = Counter(Y_test)

    class_indices = sorted(args.classes.keys())
    class_names = [args.classes[i] for i in class_indices]

    data = np.array([
        [train_counts.get(i, 0) for i in class_indices],
        [val_counts.get(i, 0) for i in class_indices],
        [test_counts.get(i, 0) for i in class_indices]
    ]).T

    split_names = ['Train', 'Validation', 'Test']

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    rects1 = ax.bar(x - width, data[:, 0], width, label=split_names[0], color=colors[0])
    rects2 = ax.bar(x, data[:, 1], width, label=split_names[1], color=colors[1])
    rects3 = ax.bar(x + width, data[:, 2], width, label=split_names[2], color=colors[2])

    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Ankle Alignment Class")
    ax.set_title("Class Distribution Accross Data Splits (Train, Validation, Test)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    def autolabel(rects, i):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)} - {((int(height) / data[:, i].sum()) * 100):.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1, 0)
    autolabel(rects2, 1)
    autolabel(rects3, 2)

    fig.tight_layout()

    plt.savefig(os.path.join(output_plots_dir, "split_label_distribution.png"))
    plt.close(fig)

def save_conf_mx_plot(args: Args, conf_mx, normalize=False):
    save_name = "conf_mx_normalized" if normalize else "conf_mx" 

    if normalize:
        conf_mx = conf_mx.astype('float') / conf_mx.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(args.classes))
    plt.xticks(tick_marks, [args.classes[i] for i in range(len(args.classes))], rotation=45)
    plt.yticks(tick_marks, [args.classes[i] for i in range(len(args.classes))])

    fmt = '.2f' if normalize else 'd'
    thresh = conf_mx.max() / 2
    for i in range(conf_mx.shape[0]):
        for j in range(conf_mx.shape[1]):
            plt.text(j, i, format(conf_mx[i, j], fmt),
                     ha="center", va="center",
                     color="white" if conf_mx[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(args.output_dir, "plots", f"{save_name}.png"))
    plt.close()

def setup_optimizer(args: Args, model):
    if args.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        args.logger.warning(f"Unknown optimizer: {args.optimizer}. Defaulting to Adam...")
        return torch.optim.Adam(model.parameters(), lr=args.lr)

def _calculate_label_weights(args: Args, train_labels):
    num_classes = len(args.classes)
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)

    weights = []
    for i, count in enumerate(class_counts):
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 1.0

        weights.append(weight)

        if args.logger:
            class_name = args.classes[i]
            args.logger.info(f"{class_name}: Count = {count}, Weight = {weight}")

    return weights

def setup_loss(args: Args, train_labels=None):
    weights = None
    if args.use_label_weights and train_labels is not None:
        args.logger.info("Calculating weights for classes.")
        weights = _calculate_label_weights(args, train_labels)
        weights = torch.tensor(weights, dtype=torch.float32).to(args.device)
    if args.loss_name.lower() == "ce":
        return torch.nn.CrossEntropyLoss(weight=weights)
    else:
        args.logger.warning(f"Unknown loss function: {args.loss_name}. Defaulting to ce (Cross Entropy).")
        return torch.nn.CrossEntropyLoss(weight=weights)