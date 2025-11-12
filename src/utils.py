import os
import logging

import matplotlib.pyplot as plt

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

def load_model(model_path: str):
    # TODO
    pass