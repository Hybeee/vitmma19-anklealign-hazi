import cv2
import numpy as np

import json
import os
import glob
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Args
import utils as utils

class_labels = {
    '1_Pronacio': 0,
    '2_Neutralis': 1,
    '3_Szupinacio': 2
}

# Egy személy esetén a label-ek így szerepelnek
wrong_labels = {
    'pronation': 0,
    'neutral': 1,
    'supination': 2
}

def _resize_with_padding(args: Args, image):
    res = args.resolution

    h, w = image.shape[:2]
    scale = min(res / h, res / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((res, res, 3), dtype=np.uint8)

    x_offset = (res - new_w) // 2
    y_offset = (res - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--logs_dir_write", type=bool, required=True)
    args_cli = parser.parse_args()
    timestamp = args_cli.timestamp
    logs_dir_write = args_cli.logs_dir_write

    args = Args()
    args.output_dir = os.path.join("outputs", f"{timestamp}_{args.model_alias}")
    output_plots_dir = os.path.join(args.output_dir, "plots")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    log_dir = args.output_dir
    args.logger = utils.get_logger(timestamp=timestamp, log_dir=log_dir, logs_dir_write=logs_dir_write)

    args.log_config()

    numpy_all_dir = os.path.join(args.data_dir, "numpy_all_data")
    os.makedirs(numpy_all_dir, exist_ok=True)

    root = os.path.join("data", "all_data")

    args.logger.info(f"Preparing data for pipeline from: {root}.")

    class_counts = {
        0: 0,
        1: 0,
        2: 0
    }

    data_count = 0

    aspect_ratios = []
    all_images = []
    all_labels = []

    for dir_name in sorted(os.listdir(root)):
        curr_dir = os.path.join(root, dir_name)
        if os.path.isdir(curr_dir):
            json_paths = glob.glob(os.path.join(curr_dir, "*.json"))
            if len(json_paths) > 0:
                json_path = json_paths[0]
            else:
                args.logger.warning(f"Skipping {dir_name}. Warning: No JSON found in directory.")
                continue

            with open(json_path, 'r') as file:
                data = json.load(file)

            for point in data:
                data_count += 1
                try:
                    image_path = "N/A"
                    image_name = point["file_upload"][9:]

                    if dir_name == "H51B9J":
                        base, ext = os.path.splitext(image_name)
                        image_name = base[:-7] + ext

                    image_path = os.path.join(curr_dir, image_name)

                    if not os.path.exists(image_path):
                        args.logger.warning(f"Skipping: {image_path}. Error: Path does not exist.")
                        continue

                    image = cv2.imread(image_path)

                    if image is None:
                        args.logger.warning(f"Skipping: {image_path}. Error: Image cannot be read, file is possibly corrupted.")
                        continue  

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    aspect_ratio = image.shape[0]/image.shape[1]

                    if args.use_padding:
                        image = _resize_with_padding(args, image)
                    else:
                        image = cv2.resize(image, (args.resolution, args.resolution))
                    image = np.array(image, dtype=np.float32)
                    image /= 255.0

                    choice = point["annotations"][0]["result"][0]["value"]["choices"][0]
        
                    try:
                        label = class_labels[choice]
                    except:
                        label = wrong_labels[choice]

                except Exception as e:
                    args.logger.warning(f"Skipping: {image_path}. Error: {e}")
                    continue

                aspect_ratios.append(aspect_ratio)
                all_images.append(image)
                all_labels.append(label)
                class_counts[label] += 1

    X = np.array(all_images)
    Y = np.array(all_labels)

    np.save(os.path.join(numpy_all_dir, "images_all.npy"), X)
    np.save(os.path.join(numpy_all_dir, "labels_all.npy"), Y)

    args.logger.info("Data Preparation Summary")
    args.logger.info(f"\tTotal data count: {data_count}")
    args.logger.info(f"\tTotal processed and saved images: {len(X)}")
    args.logger.info(f"\tImage array shape: {X.shape}")
    args.logger.info(f"\tLabel array shape: {Y.shape}")
    args.logger.info(f"Prepared data saved to: {numpy_all_dir}")

    utils.plot_label_distribution(args, class_counts, output_plots_dir)
    utils.plot_aspect_ratio(aspect_ratios, output_plots_dir)


if __name__ == "__main__":
    main()