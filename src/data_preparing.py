from PIL import Image
import numpy as np

import json
import os
import glob
import argparse

from config import Args
import utils as utils

classes = ['1_Pronacio', '2_Neutralis', '3_Szupinacio']

class_labels = {
    '1_Pronacio': 0,
    '2_Neutralis': 1,
    '3_Szupinacio': 2
}

# Egy személy esetén a label-ek így szerepelnek
wrong_labels = {
    'pronacio': 0,
    'neutralis': 1,
    'szupinacio': 2
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
        '0': 0,
        '1': 0,
        '2': 0
    }

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
                continue

            with open(json_path, 'r') as file:
                data = json.load(file)

            for point in data:
                try:
                    image_name = point["file_upload"][9:]
                    image_path = os.path.join(curr_dir, image_name)
                    image = Image.open(image_path)
                    aspect_ratios.append(image.height/image.width)
                    image = image.resize((args.resolution, args.resolution))
                    image = np.array(image)
                    image /= 255.0

                    all_images.append(image)

                    choice = point["annotations"][0]["result"][0]["value"]["choices"][0]

                    try:
                        label = class_labels[choice]
                    except:
                        label = wrong_labels[choice]

                    all_labels.append(label)

                    class_counts[str(label)] += 1

                except (KeyError, IndexError, TypeError):
                    args.logger.warning(f"\tSkipping: {image_name}")

    X = np.array(all_images)
    Y = np.array(all_labels)

    np.save(os.path.join(numpy_all_dir, "images_all.npy"), X)
    np.save(os.path.join(numpy_all_dir, "labels_all.npy"), Y)

    args.logger.info("Data Preparation Summary")
    args.logger.info(f"Total processed and saved images: {len(X)}")
    args.logger.info(f"Image array shape: {X.shape}")
    args.logger.info(f"Label array shape: {Y.shape}")

    utils.plot_label_distribution(args, class_counts, output_plots_dir)
    utils.plot_aspect_ratio(aspect_ratios, output_plots_dir)


if __name__ == "__main__":
    main()