import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from config import Args
import utils

import os
import argparse
import json

def _save_conf_mx_plot(args: Args, conf_mx, normalize=False):
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
    plt.savefig()
    plt.close(os.path.join(args.output_dir, "plots", f"{save_name}.png"))

def save_result(args: Args, result, timestamp):
    results = \
    {
        "timestamp": timestamp,
        "accuracy": result['accuracy'],
        "loss": result['loss'],
        "confusion_matrix": result['conf_mx'].tolist(),
        "hyperparameters": 
        {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "early_stopping": args.early_stopping,
            "loss_fn": str(args.f_loss),
            "device": str(args.device),
            "preprocessing_hyperparameters": 
            {
                "resolution": args.resolution,
                "color_jitter": 
                {
                    "brightness": args.cj_brightness,
                    "contrast": args.cj_contrast,
                    "saturation": args.cj_saturation,
                },
                "rotation": args.rotation,
                "normalization":
                {
                    "normalization_mean": args.norm_mean,
                    "normalization_std": args.norm_std
                }
            }
        }
    }

    with open(os.path.join(args.output_dir, "results.json"), 'w') as file:
        json.dump(results, file, indent=2)

    _save_conf_mx_plot(args, result['conf_mx'], normalize=False)
    _save_conf_mx_plot(args, result['conf_mx'], normalize=True)

def evaluate_model(args: Args, model, eval_loader):
    model.eval()
    model.to(args.device)
    # Loss
    running_loss = 0.0

    # Accuracy
    correct = 0
    total = 0

    # Confusion matrix
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in iter(eval_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = model(images)

            loss = args.f_loss(outputs, torch.argmax(labels, dim=1))
            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            _, label = torch.max(labels, dim=1)

            total += labels.size(0)
            correct += (predicted == label).sum().item()

            all_targets.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / len(eval_loader)
    accuracy = correct / total

    conf_mx = confusion_matrix(all_targets, all_predictions)

    result = {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "conf_mx": conf_mx
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp

    args = Args()
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=args.output_dir)

    model = utils.load_model()

    result = evaluate_model(arsg=args, model=model)

    save_result(args=args, result=result, timestamp=timestamp)

if __name__ == "__main__":
    main()