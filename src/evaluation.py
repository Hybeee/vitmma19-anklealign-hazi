import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from config import Args
import utils
from models import get_model, load_trained_model
from data_pipeline.data_processing import get_loader

import os
import argparse
import json

def save_result(args: Args, result, timestamp):
    results = \
    {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "classes": args.classes,
        "data_cleaning": {
            "similarity_threshold": args.similarity_threshold,
            "sharpness_threshold": args.sharpness_threshold,
            "min_contrast_std": args.min_contrast_std,
            "min_brightness_mean": args.min_brightness_mean
        },
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
        },
        "evaluation": {
            "accuracy": result['accuracy'],
            "avg_loss": result['avg_loss'],
            "confusion_matrix": result['conf_mx'].tolist(),
            "classification_report": result['classification_report']
        }
    }

    with open(os.path.join(args.output_dir, "results.json"), 'w') as file:
        json.dump(results, file, indent=2)

    utils.save_conf_mx_plot(args, result['conf_mx'], normalize=False)
    utils.save_conf_mx_plot(args, result['conf_mx'], normalize=True)

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

            loss = args.f_loss(outputs, labels)
            running_loss += loss.item()

            _, pred_class = torch.max(outputs, dim=1)
            label_class = labels

            total += labels.size(0)
            correct += (pred_class == label_class).sum().item()

            all_targets.extend(label_class.cpu().numpy())
            all_predictions.extend(pred_class.cpu().numpy())
    
    avg_loss = running_loss / len(eval_loader)
    accuracy = correct / total

    target_names = [args.classes[i] for i in range(len(args.classes))]
    conf_mx = confusion_matrix(all_targets, all_predictions)
    report = classification_report(y_true=all_targets,
                                   y_pred=all_predictions,
                                   target_names=target_names,
                                   digits=4,
                                   output_dict=True)

    result = {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "conf_mx": conf_mx,
        "classification_report": report
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp

    args = Args()
    args.output_dir = os.path.join("outputs", timestamp)

    log_dir = args.output_dir
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=log_dir)

    train_labels = None

    if args.model_name.lower() == "dummy_baseline":
        train_labels = np.load(os.path.join(args.data_dir, "splits", "train_labels.npy"))
    
    model = load_trained_model(args, train_labels=train_labels)

    test_images = np.load(os.path.join(args.data_dir, "splits", "test_images.npy"))
    test_labels = np.load(os.path.join(args.data_dir, "splits", "test_labels.npy"))

    test_loader = get_loader(args, "test",
                             images=test_images, labels=test_labels)

    args.logger.info(f"Running evaluation on model: {args.model_name}")

    result = evaluate_model(args=args, model=model, eval_loader=test_loader)

    args.logger.info(f"Saving evaluation results and pipeline data to: {args.output_dir}\\results.json")

    save_result(args=args, result=result, timestamp=timestamp)

if __name__ == "__main__":
    main()