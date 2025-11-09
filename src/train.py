import numpy as np
import torch

import os
from datetime import datetime

def train(args, model):
    logger = args.logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_loader, val_loader = get_loader(args) # TODO: implement, will be in data_utils.py probably

    device = args.device
    optimizer = args.optimizer
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

            loss = f_loss(outputs, labels)
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

                loss = f_loss(outputs, labels)
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

            logger.info(f"Validation loss has decreased. Saving model to: {args.model_dir}")

            model.to("cpu")
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
            
            torch.save(model, os.path.join(args.model_dir, f"model_{timestamp}.pth"))
            model.to(device)
        else:
            no_improvement += 1
            logger.info(f"Validation loss has not decreased.")
        
        if no_improvement == args.early_stopping:
            logger.info(f"Early stopping threshold reached. Training completed.")
            save_plots(args, train_loss, train_accuracy, val_loss, val_accuracy) # TODO: implement
            return
    
    logger.info("Training completed.")
