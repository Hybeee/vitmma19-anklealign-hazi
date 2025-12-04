import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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

    return unique_images, unique_labels

def calculate_quality_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    contrast_std = np.std(gray)

    mean_intensity = np.mean(gray)

    return laplacian_var, contrast_std, mean_intensity

def exclude_low_quality_images(args: Args, images, labels, output_plots_dir):

    images_to_keep = []
    labels_to_keep = []
    total_excluded = 0

    for idx, img in enumerate(images):
        # NOTE: A képek [0, 1] normalizáltak ezen a ponton. A következő számításhoz
        # célszerű [0, 255]-ös képekkel dolgozni.
        img_255 = (img * 255).astype(np.uint8)

        sharpness, contrast, mean_intensity = calculate_quality_metrics(img_255)

        reason = ""

        if sharpness < args.sharpness_threshold:
            reason += f"sharpness (var: {sharpness:.1f} < {args.sharpness_threshold})"
        if contrast < args.min_contrast_std:
            reason += f"\nLow contast (std: {contrast:.1f} < {args.min_contrast_std})"
        if mean_intensity < args.min_brightness_mean:
            reason += f"\nLow brightness (std: {mean_intensity:.1f} < {args.min_brightness_mean})"
        
        if reason != "":
            total_excluded += 1
            args.logger.warning(f"Excluding image. Reason: {reason}")
            if args.save_lq:
                utils.save_lq(img, idx, reason, output_plots_dir)
        else:
            images_to_keep.append(img)
            labels_to_keep.append(labels[idx])

    args.logger.info(f"Finished quality control. Total images excluded: {total_excluded}")

    return np.array(images_to_keep), np.array(labels_to_keep)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp

    args = Args()
    args.output_dir = os.path.join("outputs", f"{timestamp}_{args.model_alias}")
    output_plots_dir = os.path.join(args.output_dir, "plots")

    log_dir = args.output_dir
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=log_dir)

    numpy_all_dir = os.path.join(args.data_dir, "numpy_all_data")
    
    images = np.load(os.path.join(numpy_all_dir, "images_all.npy"))
    labels = np.load(os.path.join(numpy_all_dir, "labels_all.npy"))

    images, labels = exclude_duplicates(args, images, labels, output_plots_dir)
    images, labels = exclude_low_quality_images(args, images, labels, output_plots_dir)

    cleaned_np_dir = os.path.join(args.data_dir, "cleaned_numpy_data")
    os.makedirs(cleaned_np_dir, exist_ok=True)

    args.logger.info("Finished data cleaning")
    args.logger.info(f"Remaining images shape: {images.shape}")
    args.logger.info(f"Remaining labels shape: {labels.shape}")

    np.save(os.path.join(cleaned_np_dir, "images.npy"), images)
    np.save(os.path.join(cleaned_np_dir, "labels.npy"), labels)

if __name__ == "__main__":
    main()