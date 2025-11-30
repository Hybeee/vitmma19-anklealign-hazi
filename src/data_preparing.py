import cv2
import numpy as np

import json
import os
import glob
import argparse

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
    os.makedirs(numpy_all_dir, exist_ok=True)

    root = os.path.join("data", "all_data")

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
    args.logger.info(f"Total data count: {data_count}")
    args.logger.info(f"Total processed and saved images: {len(X)}")
    args.logger.info(f"Image array shape: {X.shape}")
    args.logger.info(f"Label array shape: {Y.shape}")

    utils.plot_label_distribution(args, class_counts, output_plots_dir)
    utils.plot_aspect_ratio(aspect_ratios, output_plots_dir)


if __name__ == "__main__":
    main()