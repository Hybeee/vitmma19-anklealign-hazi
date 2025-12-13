import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import Args
import utils
from models import load_trained_model
from data_pipeline.data_processing import get_loader

import os
import argparse
import json
import pprint

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
            "loss_fn": str(args.loss_name),
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

def evaluate_model(args: Args, model, eval_loader, f_loss):
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
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = f_loss(outputs, labels)
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

    args.logger.info("Evaluation results")
    args.logger.info(f"Average loss: {avg_loss:.4f}")
    args.logger.info(f"Accuracy: {accuracy:.4f}")
    args.logger.info(f"Confusion matrix:\n{pprint.pformat(conf_mx)}")
    args.logger.info(f"Classification report (per-class metrics):\n{pprint.pformat(report, indent=4)}")

    return result

def run_gradcam_analysis(args: Args, model, images, labels, num_samples=5):
    args.logger.info(f"Starting Grad-CAM analysis on {num_samples} random images...")

    target_layers = []
    if args.model_name.lower() == "anklealign_simple":
        target_layers = [model.conv4]
    elif args.model_name.lower() == "anklealign_medium":
        target_layers = [model.block4[0]]
    elif args.model_name.lower() == "anklealign_complex":
        target_layers = [model.layer4[-1]]
    else:
        args.logger.warning(f"Grad-CAM is not configured for model: {args.model_name}")
        return
    
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    ])

    if len(images) < num_samples:
        num_samples = len(images)

    indices = np.random.choice(len(images), num_samples, replace=False)

    model.to(args.device)
    model.eval()
    cam = GradCAM(model=model, target_layers=target_layers)

    save_path = os.path.join(args.output_dir, "plots", "gradcam")
    os.makedirs(save_path, exist_ok=True)

    for i, idx in enumerate(indices):
        raw_image = images[idx]
        true_label_idx = labels[idx]
        true_label_name = args.classes[true_label_idx]

        input_tensor = preprocess_transform(raw_image).unsqueeze(0).to(args.device)

        output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        pred_idx = pred_idx.item()
        pred_label_name = args.classes[pred_idx]

        targets = [ClassifierOutputTarget(pred_idx)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(raw_image, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(raw_image)
        plt.title(f"True: {true_label_name}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Pred: {pred_label_name}\n(Grad-CAM)")
        plt.axis('off')

        plt.savefig(os.path.join(save_path, f"gradcam_{i}.png"), bbox_inches='tight')
        plt.close()
    
    args.logger.info(f"Grad-CAM visualizations saved to: {save_path}")

def create_attention_maps(args: Args, model, images, labels, num_samples=5):
    args.logger.info(f"Visualizing attention maps.")

    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    ])

    if len(images) < num_samples:
        num_samples = len(images)

    indicies = np.random.choice(len(images), num_samples, replace=False)

    model.to(args.device)
    model.eval()

    save_path = os.path.join(args.output_dir, "plots", "attention_maps")
    os.makedirs(save_path, exist_ok=True)

    for i, idx in enumerate(indicies):
        raw_image = images[idx]
        true_label_idx = labels[idx]
        true_label_name = args.classes[true_label_idx]

        input_tensor = preprocess_transform(raw_image).unsqueeze(0).to(args.device)

        (output, att_mat) = model(input_tensor)

        _, pred_idx = torch.max(output, 1)
        pred_idx = pred_idx.item()
        pred_label_name = args.classes[pred_idx]

        att_mat = torch.stack(att_mat).squeeze(1)
        att_mat = torch.mean(att_mat, dim=1)

        residual_att = torch.eye(att_mat.size(1)).to(args.device)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        joint_attentions = torch.zeros(aug_att_mat.size(), device=args.device)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        mask = cv2.resize(mask / mask.max(), (raw_image.shape[1], raw_image.shape[0]))

        raw_image = (raw_image * 255).astype(np.uint8)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(raw_image)
        plt.title(f"True: {true_label_name}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(raw_image)
        plt.imshow(mask, cmap='jet', alpha=0.5)
        plt.title(f"Pred: {pred_label_name}\n(Attention Map)")
        plt.axis('off')

        plt.savefig(os.path.join(save_path, f"attention_map_{i}.png"), bbox_inches='tight')
        plt.close()

    args.logger.info(f"Attention Map visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp

    args = Args()
    args.output_dir = os.path.join("outputs", f"{timestamp}_{args.model_alias}")
    output_plots_dir = os.path.join(args.output_dir, "plots")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    log_dir = args.output_dir
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=log_dir)

    train_labels = None

    train_labels = np.load(os.path.join(args.data_dir, "splits", "train_labels.npy"))

    model = load_trained_model(args, train_labels=train_labels)

    test_images = np.load(os.path.join(args.data_dir, "splits", "test_images.npy"))
    test_labels = np.load(os.path.join(args.data_dir, "splits", "test_labels.npy"))

    test_loader = get_loader(args, "test",
                             images=test_images, labels=test_labels)

    args.logger.info(f"Running evaluation on model: {args.model_name}")

    f_loss = utils.setup_loss(args=args, train_labels=train_labels)

    result = evaluate_model(args=args, model=model, eval_loader=test_loader, f_loss=f_loss)

    args.logger.info(f"Saving evaluation results and pipeline data to: {os.path.join(args.output_dir, 'results.json')}")

    save_result(args=args, result=result, timestamp=timestamp)

    try:
        if args.model_name == "anklealign_vit":
            create_attention_maps(args=args, model=model,
                                  images=test_images, labels=test_labels)
        else:
            run_gradcam_analysis(args=args, model=model,
                                 images=test_images, labels=test_labels)
    except Exception as e:
        args.logger.error(f"Failed to run model explainability visualization for {args.model_name}. Reason: {e}")

if __name__ == "__main__":
    main()