import numpy as np
import torch
import matplotlib.pyplot as plt

import os
from datetime import datetime
import argparse

from config import Args as Args
from data_processing import get_loader
import utils as utils
from models import get_model, DummyBaseLine

def setup_optimizer(args, model):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)

def _evaluate_dummy(args: Args, model: DummyBaseLine, data_loader, split_name="train"):
    f_loss = args.f_loss
    
    running_loss, running_accuracy = 0.0, 0.0
    total, correct = 0, 0

    model.to(args.device)
    model.eval()

    for images, labels in iter(data_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)

        outputs = model(images)

        loss = f_loss(outputs, labels)
        running_loss += loss.item()

        _, pred_class = torch.max(outputs, 1)
        label_class = labels
        total += labels.size(0)
        correct += (pred_class == label_class).sum().item()

    
    running_loss /= len(data_loader)
    running_accuracy = 100 * correct / total

    args.logger.info(f"Dummy Baseline {split_name} loss: {running_loss}")
    args.logger.info(f"Dummy Baseline {split_name} accuracy: {running_accuracy}")

    running_loss, running_accuracy = 0.0, 0.0
    total, correct = 0, 0

def train(args: Args, train_loader, val_loader, model, optimizer):

    if args.model_name.lower() == "dummy_baseline":
        args.logger.info("Evaluating Dummy Baseline on train and test sets")
        _evaluate_dummy(args, model, train_loader, "train")
        args.logger.info("Note: Training accuracy might differ from accuracy on outputs/timestamp/plots/split_class_distribution.png." \
        " Reason: train_loader has drop_last=True.")
        _evaluate_dummy(args, model, val_loader, "validation")
        return

    logger = args.logger

    device = args.device
    f_loss = args.f_loss

    min_loss = np.inf
    no_improvement = 0

    train_loss = list()
    train_accuracy = list()

    val_loss = list()
    val_accuracy = list()

    for e in range(args.epochs):
        running_loss, running_accuracy = 0.0, 0.0
        total, correct = 0, 0

        model.to(device)
        model.train()

        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = f_loss(outputs, torch.argmax(labels, dim=1))
            running_loss += loss.item()

            _, pred_class = torch.max(outputs, 1)
            _, label_class = torch.max(labels, 1)
            total += labels.size(0)
            correct += (pred_class == label_class).sum().item()

            loss.backward()
            optimizer.step()
        
        running_loss /= len(train_loader)
        running_accuracy = 100 * correct / total

        logger.info(f"Epoch = {e}")
        logger.info(f"\tTraining loss = {running_loss}")
        logger.info(f"\tTraining accuracy = {running_accuracy}")

        train_loss.append(running_loss)
        train_accuracy.append(running_accuracy)

        running_loss, running_accuracy = 0.0, 0.0
        total, correct = 0, 0

        model.eval()

        with torch.no_grad():
            for images, labels in iter(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = f_loss(outputs, torch.argmax(labels, dim=1)) # f_loss will be torch.nn.CrossEntropy
                running_loss += loss.item()

                _, pred_class = torch.max(outputs, 1)
                _, label_class = torch.max(labels, 1)
                total += labels.size(0)
                correct += (pred_class == label_class).sum().item()
        
            running_loss /= len(val_loader)
            running_accuracy = 100 * correct / total

            logger.info(f"\tValidation loss = {running_loss}")
            logger.info(f"\tValidation accuracy = {running_accuracy}")

            val_loss.append(running_loss)
            val_accuracy.append(running_accuracy)

        if val_loss[-1] < min_loss:
            min_loss = val_loss[-1]
            no_improvement = 0

            logger.info(f"Validation loss has decreased. Saving model to: {args.output_dir}")

            model.to("cpu")
            
            torch.save(model, os.path.join(args.output_dir, f"model.pth"))
            model.to(device)
        else:
            no_improvement += 1
            logger.info(f"Validation loss has not decreased.")
        
        if no_improvement == args.early_stopping:
            logger.info(f"Early stopping threshold reached. Training completed.")
            utils.save_plots(args, train_loss, train_accuracy, val_loss, val_accuracy)
            return
    
    logger.info("Training completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp

    args = Args()
    args.output_dir = os.path.join("outputs", timestamp)

    log_dir = args.output_dir
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=log_dir)

    train_images = np.load(os.path.join(args.data_dir, "splits", "train_images.npy"))
    train_labels = np.load(os.path.join(args.data_dir, "splits", "train_labels.npy"))
    val_images = np.load(os.path.join(args.data_dir, "splits", "val_images.npy"))
    val_labels = np.load(os.path.join(args.data_dir, "splits", "val_labels.npy"))

    model = get_model(args, train_labels=train_labels)

    optimizer = setup_optimizer(args=args, model=model)

    train_loader = get_loader(args, "train",
                              images=train_images, labels=train_labels)
    val_loader = get_loader(args, "val",
                            images=val_images, labels=val_labels)

    train(args=args,
          train_loader=train_loader, val_loader=val_loader,
          model=model, optimizer=optimizer)
    
if __name__ == "__main__":
    main()