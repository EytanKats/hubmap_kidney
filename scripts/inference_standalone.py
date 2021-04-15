import os
import gc
import cv2
import pathlib
import rasterio

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm
from rasterio.windows import Window
from simple_converge.metrics.metrics import dice, precision, recall


TRAIN_DATA_DIR = "/data/eytank/datasets/hubmap_kidney/raw_data/images/train/"
MODEL_DIR = "/data/eytank/simulations/hubmap_kidney/2021.03.27_smefficientnetb4_updateddata/0/model/"
MASKS_CSV_FILE_PATH = "/data/eytank/datasets/hubmap_kidney/raw_data/annotations/train/train.csv"
RESULTS_FILE_PATH = "/data/eytank/simulations/hubmap_kidney/2021.03.27_smefficientnetb4_updateddata/0/scores_512_512_tta.csv"
IMAGES_TO_TEST = ["0486052bb.tiff", "095bf7a1f.tiff", "1e2425f28.tiff"]  # fold 0

CHOOSE_RANDOM_TILES = False
RANDOM_TILES_NUM = 100
PLOT_TILE_PREDICTIONS = False

TILE_SIZE = (1024, 1024)
TILE_STEP = (512, 512)

TTA_ROTATION = True

PREDICTION_THRESHOLD = 0.5
AGGREGATION_THRESHOLD = 0.5

APPLY_PREDICTION_FILTER = True
prediction_filter = np.zeros(shape=TILE_SIZE, dtype=np.uint8)
prediction_filter[256:768, 256:768] = 1


def make_grid(shape, window_x, window_y, step_x, step_y):

    xs = np.arange(0, shape[0] - window_x, step=step_x)
    ys = np.arange(0, shape[1] - window_y, step=step_y)

    slices = np.zeros((len(xs), len(ys), 4), dtype=np.int32)
    for x_idx, x_left in enumerate(xs):
        for y_idx, y_top in enumerate(ys):
            slices[x_idx, y_idx] = x_left, x_left + window_x, y_top, y_top + window_y

    return slices.reshape(len(xs) * len(ys), 4)


def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):

        if isinstance(enc, np.float) and np.isnan(enc):
            continue

        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1 + m

    return img.reshape(shape).T


# Load model
print("\nLoading model {0}".format(os.path.basename(MODEL_DIR)))
model = tf.keras.models.load_model(MODEL_DIR)

# Load annotations file
df_masks = pd.read_csv(MASKS_CSV_FILE_PATH).set_index('id')

# Initialize metrics placeholders
dice_scores = list()
precision_scores = list()
recall_scores = list()

# Iterate other images and make predictions
for img_name in IMAGES_TO_TEST:

    # Read image and corresponding mask
    img_data = rasterio.open(os.path.join(TRAIN_DATA_DIR, img_name), transform=rasterio.Affine(1, 0, 0, 0, 1, 0))
    mask = enc2mask(df_masks.loc[pathlib.Path(img_name).stem], (img_data.shape[1], img_data.shape[0]))

    if img_data.count != 3:
        subdatasets = img_data.subdatasets
        layers = []
        if len(subdatasets) > 0:
            for i, subdataset in enumerate(subdatasets, 0):
                layers.append(rasterio.open(subdataset))

    # Make tiles grid
    tiles = make_grid(img_data.shape, TILE_SIZE[0], TILE_SIZE[1], TILE_STEP[0], TILE_STEP[1])

    # Choose random tile indices and make predictions for chosen tiles
    if CHOOSE_RANDOM_TILES:
        tiles = [tiles[idx] for idx in np.random.choice(np.arange(0, len(tiles)), RANDOM_TILES_NUM)]

    # Initialize predictions mask and run model on tiles
    predictions_mask = np.zeros(img_data.shape, dtype=np.uint8)
    for (x1, x2, y1, y2) in tqdm(tiles, desc=img_name):

        # Crop tile from the image
        if img_data.count == 3:
            img = img_data.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))
            img = np.moveaxis(img, 0, -1)
        else:
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)
            for i, layer in enumerate(layers):
                img[:, :, i] = layer.read(1, window=Window.from_slices((x1, x2), (y1, y2)))

        # Crop tile from the mask
        mask_tile = mask[x1:x2, y1:y2]

        # Preprocess image
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pre_bgr_img = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        pre_bgr_img = pre_bgr_img.astype(np.float32)
        pre_bgr_img = tf.keras.applications.resnet50.preprocess_input(pre_bgr_img)

        # Apply test time augmentation
        if TTA_ROTATION:
            pre_bgr_img_tta_list = [pre_bgr_img]
            for rot in [1, 2, 3]:
                pre_bgr_img_tta_list.append(np.rot90(pre_bgr_img, k=rot, axes=(0, 1)))
            pre_bgr_img = np.array(pre_bgr_img_tta_list)
        else:
            pre_bgr_img = np.expand_dims(pre_bgr_img, 0)

        # Predict
        bgr_pred = model.predict(pre_bgr_img)

        # Apply test time augmentation
        if TTA_ROTATION:
            bgr_pred_tta_sum = bgr_pred[0, ...]
            for rot in [1, 2, 3]:
                bgr_pred_tta_sum += np.rot90(bgr_pred[rot, ...], k=rot, axes=(1, 0))
            bgr_pred = bgr_pred_tta_sum / 4
        else:
            bgr_pred = np.squeeze(bgr_pred)

        # Postprocess prediction
        bgr_pred = (bgr_pred > PREDICTION_THRESHOLD).astype(np.uint8)
        bgr_pred = cv2.resize(bgr_pred, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        if APPLY_PREDICTION_FILTER:
            bgr_pred *= prediction_filter

        # Fill appropriate region in predictions mask
        predictions_mask[x1:x2, y1:y2] += bgr_pred

        # Plot tile predictions
        if PLOT_TILE_PREDICTIONS:
            fig = plt.figure(figsize=(10, 10))

            fig.add_subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(img)
            plt.imshow(bgr_pred, alpha=0.5)

            fig.add_subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(mask_tile)
            plt.imshow(bgr_pred, alpha=0.5)

            plt.savefig()

    del img, bgr_img, pre_bgr_img, mask_tile, bgr_pred
    gc.collect()

    # Postprocess predictions mask
    predictions_mask = (predictions_mask >= AGGREGATION_THRESHOLD).astype(np.uint8)

    # Calculate metrics
    print("Calculating metrics for image {0}".format(img_name))
    dice_for_image = dice(predictions_mask, mask)
    precision_for_image = precision(predictions_mask, mask)
    recall_for_image = recall(predictions_mask, mask)

    dice_scores.append(dice_for_image)
    precision_scores.append(precision_for_image)
    recall_scores.append(recall_for_image)

    print(" - dice = {0}".format(dice_for_image))
    print(" - precision = {0}".format(precision_for_image))
    print(" - recall = {0}".format(recall_for_image))

    del mask, predictions_mask, img_data, tiles
    gc.collect()

# Fill CSV file with metrics and save it
df_results = pd.DataFrame()
df_results["id"] = IMAGES_TO_TEST
df_results["dice"] = dice_scores
df_results["precision"] = precision_scores
df_results["recall"] = recall_scores
df_results.to_csv(os.path.join(RESULTS_FILE_PATH), index=False)

del model, df_results
gc.collect()

print("End of script")
exit()
