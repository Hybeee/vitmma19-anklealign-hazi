import numpy as np
import matplotlib.pyplot as plt

import os
import argparse

import utils as utils
from config import Args

def are_images_identical(args: Args, img1, img2):
    if img1 is None:
        args.logger.error("First image is None. Returning False.")
        return False
    if img2 is None:
        args.logger.error("Second image is None. Returning False.")
        return False
    
    if img1.shape != img2.shape:
        args.logger.warning("Images have different shapes. Returning False")
        return False
    
    return np.array_equal(img1, img2)

def are_images_identical_distance(args: Args, img1, img2):
    if img1 is None:
        args.logger.error("First image is None. Returning False.")
        return False
    if img2 is None:
        args.logger.error("Second image is None. Returning False.")
        return False
    
    if img1.shape != img2.shape:
        args.logger.warning("Images have different shapes. Returning False")
        return False
    
    diff = img1.astype('float') - img2.astype('float')

    mse = np.sum(diff ** 2) / np.prod(img1.shape)

    if mse < args.similarity_threshold:
        return True
    else:
        return False

def exclude_duplicates(args: Args, images, labels, output_plots_dir):
    keep = [True] * len(images)
    duplicate_count = 0
    duplicate_indices = []

    for i in range(len(images)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(images)):
            if not keep[j]:
                continue

            if are_images_identical_distance(args, images[i], images[j]):
                keep[j] = False
                duplicate_count += 1
                duplicate_indices.append([i, j])

                if args.save_dupes:
                    utils.save_dupes(images, i, j, output_plots_dir)

    unique_images = [images[i] for i in range(len(images)) if keep[i]]
    unique_labels = [labels[i] for i in range(len(labels)) if keep[i]]

    args.logger.info(f"Total duplicates found: {duplicate_count}.")
    args.logger.info(f"\tDuplicate indices: {duplicate_indices}")

    return unique_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp

    args = Args()
    args.output_dir = os.path.join("outputs", timestamp)
    output_plots_dir = os.path.join(args.output_dir, "plots")

    log_dir = args.output_dir
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=log_dir)

    numpy_all_dir = os.path.join(args.data_dir, "numpy_all_data")
    
    images = np.load(os.path.join(numpy_all_dir, "images_all.npy"))
    labels = np.load(os.path.join(numpy_all_dir, "labels_all.npy"))

    images, labels = exclude_duplicates(args, images, labels, output_plots_dir)

if __name__ == "__main__":
    main()