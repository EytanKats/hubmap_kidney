import os
import cv2
import glob
import pathlib
import numpy as np
import pandas as pd
import tifffile as tiff

from plots.plots import overlay_plot
from metrics.metrics import dice, precision, recall
from utils.dataset_utils import load_dataset_file, rle_to_mask, mask_to_rle

raw_images_dir = "../../Datasets/HuBMAP_Kidney/raw_data/images/train"
rle_encodings_path = "../../Datasets/HuBMAP_Kidney/raw_data/annotations/train/train.csv"
prediction_template = "../../Datasets/HuBMAP_Kidney/masks_1000/*.png"
output_dir = "../../Datasets/HuBMAP_Kidney/test"

test = False

prediction_paths = glob.glob(prediction_template)
prediction_names = [pathlib.Path(prediction_path).stem for prediction_path in prediction_paths]

# If in test mode load RLE encodings of masks in training data
if test:
    rle_encodings_gt = load_dataset_file(rle_encodings_path).set_index("id")
    dice_scores = list()
    precision_scores = list()
    recall_scores = list()
else:
    rle_encodings_result = list()

names = list()
df_results = pd.DataFrame()

original_names = np.unique([prediction_name.split('_')[0] for prediction_name in prediction_names])
for name in original_names:

    # Read original image
    raw_image_path = os.path.join(raw_images_dir, name + ".tiff")
    image = tiff.imread(raw_image_path)

    if len(image.shape) == 5:
        image = image.squeeze()
        image = np.transpose(image, (1, 2, 0))

    # Create mask placeholder
    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

    # Get prediction_paths for the image
    image_prediction_paths = [prediction_path for prediction_path in prediction_paths if name in prediction_path]

    # Create full-size mask
    for image_prediction_path in image_prediction_paths:

        prediction_mask = cv2.imread(image_prediction_path, cv2.IMREAD_GRAYSCALE)

        image_prediction_name = pathlib.Path(image_prediction_path).stem
        x = int(image_prediction_name.split('_')[1])
        y = int(image_prediction_name.split('_')[2])

        mask[y: y + 1000, x: x + 1000] = prediction_mask

    # Resize image and mask for visualization
    small_image = cv2.resize(image, (5000, 5000), interpolation=cv2.INTER_LINEAR)
    small_image = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)

    small_mask = cv2.resize(mask, (5000, 5000), interpolation=cv2.INTER_LINEAR)

    overlays = [small_mask]
    colors = [2]

    # If in test mode create ground truth mask from RLE encoding and calculate metrics
    # Else encode mask to RLE
    if test:
        gt_mask = rle_to_mask(rle_encodings_gt.loc[name]["encoding"], (image.shape[0], image.shape[1]))

        dice_scores.append(dice(mask / 255, gt_mask))
        precision_scores.append(precision(mask / 255, gt_mask))
        recall_scores.append(recall(mask / 255, gt_mask))

        small_gt_mask = cv2.resize(gt_mask, (5000, 5000), interpolation=cv2.INTER_LINEAR)
        small_gt_mask = small_gt_mask * 255

        overlays.append(small_gt_mask)
        colors.append(1)
    else:
        rle_encodings_result.append(mask_to_rle(mask))

    names.append(name)

    output_path = os.path.join(output_dir, name + ".png")
    overlay_plot(small_image, overlays, colors, outputs_path=output_path)

    # If in test mode populate dataframe with metrics
    # Else populate dataset with encodings
    df_results["id"] = names
    if test:
        df_results["dice"] = dice_scores
        df_results["precision"] = precision_scores
        df_results["recall"] = recall_scores
    else:
        df_results["encoding"] = rle_encodings_result

output_df_path = os.path.join(output_dir, "results.csv")
df_results.to_csv(output_df_path, index=False)

print("Image {0} processed".format(name))

print("End of script")