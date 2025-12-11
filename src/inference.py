import numpy as np
import torch
from torchvision import transforms

from config import Args
import utils
from models import load_trained_model

import os
import argparse

def predict(args: Args, model, images, labels, n_samples=5):
    args.logger.info(f"Running inference on {n_samples} images...")

    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    ])

    if len(images) < n_samples:
        n_samples = len(images)

    indices = np.random.choice(len(images), n_samples, replace=False)
    
    model.to(args.device)
    model.eval()

    args.logger.info("Result {i}: PREDICTED_CLASS - TRUE CLASS | CONFIDENCE | STATUS")
    for idx, i in enumerate(indices):
        image = images[i]
        true_label_index = labels[i]
        true_label_name = args.classes[true_label_index]

        input_tensor = preprocess_transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            _, predicted_index = torch.max(outputs, 1)
            predicted_index = predicted_index.item()
            predicted_class = args.classes[predicted_index]
            confidence = probabilities[0][predicted_index].item() * 100

        status = "PASS" if predicted_index == true_label_index else "FAIL"

        args.logger.info(f"Result {idx+1}: {predicted_class} - {true_label_name} | {confidence:.2f} | {status}")



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

    predict(args=args,
            model=model,
            images=test_images, labels=test_labels)


if __name__ == "__main__":
    main()