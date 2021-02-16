import os
import gc
import cv2
import glob
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile as tiff

from plots.plots import overlay_plot
from metrics.metrics import dice, precision, recall
from utils.dataset_utils import rle_to_mask, mask_to_rle, load_dataset_file

test = True

tiff_images_template = "/data/eytank/datasets/hubmap_kidney/raw_data/images/train/095bf7a1f*"
rle_encodings_path = "/data/eytank/datasets/hubmap_kidney/raw_data/annotations/train/train.csv"
model_dir = "/data/eytank/simulations/hubmap_kidney/2021.01.29_sm_unet_efficientnetb4_pretrained/1/model/"

# anatomical_structure_suffix = "-anatomical-structure.json"

tiles_dir = "/data/eytank/simulations/hubmap_kidney/2021.01.29_sm_unet_efficientnetb4_pretrained/1/inference_tiles"
predictions_dir = "/data/eytank/simulations/hubmap_kidney/2021.01.29_sm_unet_efficientnetb4_pretrained/1/inference_predictions"
results_dir = "/data/eytank/simulations/hubmap_kidney/2021.01.29_sm_unet_efficientnetb4_pretrained/1/inference_results"


submission_file_path = "/data/eytank/simulations/hubmap_kidney/2021.01.29_sm_unet_efficientnetb4_pretrained/1/inference_results/submission.csv"
scores_file_path = "/data/eytank/simulations/hubmap_kidney/2021.01.29_sm_unet_efficientnetb4_pretrained/1/inference_results/scores.csv"

tile_size = 1024
tile_step = 1024
resize_shape = (256, 256)
sat_thr = 40
num_pixels_thr = 10000

prediction_batch_size = 64


def remove_holes(binary_mask):

    postprocessed_mask = np.copy(binary_mask)
    im_floodfill = np.copy(postprocessed_mask)

    h, w = postprocessed_mask.shape[:2]
    binary_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, binary_mask, (0, 0), 2)
    cv2.floodFill(im_floodfill, binary_mask, (0, resize_shape[0] - 1), 2)
    cv2.floodFill(im_floodfill, binary_mask, (resize_shape[1] - 1, 0), 2)
    cv2.floodFill(im_floodfill, binary_mask, (resize_shape[1] - 1, resize_shape[0] - 1), 2)

    postprocessed_mask[im_floodfill != 2] = 1

    del im_floodfill
    del binary_mask
    gc.collect()

    return postprocessed_mask


print("\nCreating test dataset from raw data")

# Create tiles temporary dir
if not os.path.exists(tiles_dir):
    os.makedirs(tiles_dir)

# Load ground truth RLE encodings
if test:
    rle_encodings_gt = load_dataset_file(rle_encodings_path).set_index("id")
    dice_scores = list()
    precision_scores = list()
    recall_scores = list()

# Create test dataset
inference_images_dir = os.path.dirname(tiff_images_template)
inference_images_paths = glob.glob(os.path.join(tiff_images_template))
inference_images_names = list()
tiles_paths = list()
for inference_image_path in inference_images_paths:

    # Read image
    print(" - loading image {0}".format(os.path.basename(inference_image_path)))
    image_name = pathlib.Path(inference_image_path).stem
    inference_images_names.append(image_name)

    image = tiff.imread(inference_image_path)

    if len(image.shape) == 5:
        image = image.squeeze()
        image = np.transpose(image, (1, 2, 0))

    # Create tiles
    print(" - creating tiles for image {0} with shape {1}".format(os.path.basename(inference_image_path), image.shape))
    cur_num_tiles = len(tiles_paths)
    for x in range(0, image.shape[1] - tile_size, tile_step):
        for y in range(0, image.shape[0] - tile_size, tile_step):

            tile_image = image[y: y + tile_size, x: x + tile_size]

            # Check saturation threshold
            hsv_tile_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_tile_image)
            if (s > sat_thr).sum() > num_pixels_thr:
                output_path_tile = os.path.join(tiles_dir, image_name + "_" + str(x) + "_" + str(y) + ".png")
                tiles_paths.append(output_path_tile)

                tile_image = cv2.resize(tile_image, resize_shape, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(output_path_tile, tile_image)

            del tile_image
            del hsv_tile_image
            del h
            del s
            del v

    del image
    gc.collect()

    print(" - number of tiles for image {0} is {1}".format(os.path.basename(inference_image_path), len(tiles_paths) - cur_num_tiles))

print("\nLoading model {0}".format(os.path.basename(model_dir)))
model = tf.keras.models.load_model(model_dir)

print("\nPredicting glomeruli regions")
print(" - batch size is {0}, number of batches is {1}".format(prediction_batch_size, len(tiles_paths) // prediction_batch_size + 1))

# Create predictions directory
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

predictions_paths = list()
for batch in range(len(tiles_paths) // prediction_batch_size + 1):

    print(" - loading batch {0} data".format(batch))
    batch_data = list()
    batch_tiles_paths = tiles_paths[batch * prediction_batch_size: (batch + 1) * prediction_batch_size]
    for tile_path in batch_tiles_paths:
        tile = cv2.imread(tile_path)
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        tile = tile.astype(np.float32)
        tile = tf.keras.applications.resnet50.preprocess_input(tile)

        # Test time augmentation (TTA)
        # tile_90 = np.rot90(tile, k=1, axes=(0, 1))
        # tile_180 = np.rot90(tile, k=2, axes=(0, 1))
        # tile_270 = np.rot90(tile, k=3, axes=(0, 1))

        batch_data.append(tile)
        # batch_data.append(tile_90)
        # batch_data.append(tile_180)
        # batch_data.append(tile_270)

    batch_data = np.array(batch_data)

    print(" - predicting batch {0}, batch shape is {1}".format(batch, batch_data.shape))
    predictions = model.predict(np.array(batch_data))

    print(" - postprocessing predictions for batch {0}".format(batch))
    #     for prediction_idx in range(len(predictions) // 4):
    for prediction_idx in range(len(predictions)):
        # Merge TTA predictions
        # post_processed_prediction = predictions[prediction_idx * 4][..., 0]
        # for tta_idx in range(1, 4):
        #     tta_prediction = predictions[prediction_idx * 4 + tta_idx][..., 0]
        #     tta_prediction = np.rot90(tta_prediction, k=tta_idx, axes=(1, 0))
        #     post_processed_prediction += tta_prediction
        #
        # post_processed_prediction = post_processed_prediction / 4

        post_processed_prediction = predictions[prediction_idx][..., 0]

        # Apply threshold on predicted mask
        _, post_processed_prediction = cv2.threshold(post_processed_prediction, 0.5, 1, cv2.THRESH_BINARY)

        # Fill connected components
        post_processed_prediction = remove_holes(post_processed_prediction)

        # Save postprocessed predictions
        output_path_prediction = os.path.join(predictions_dir, os.path.basename(batch_tiles_paths[prediction_idx]))
        predictions_paths.append(output_path_prediction)

        cv2.imwrite(output_path_prediction, post_processed_prediction)

        del post_processed_prediction

    del batch_data
    del predictions
    gc.collect()

# Create results dir
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("\nMerging glomeruli predictions to get full size segmentation mask")
ids = list()
encodings = list()
for inference_image_name in inference_images_names:

    # Read original image
    print(" - loading image {0}".format(inference_image_name))
    raw_image_path = os.path.join(inference_images_dir, inference_image_name + ".tiff")
    image = tiff.imread(raw_image_path)

    if len(image.shape) == 5:
        image = image.squeeze()
        image = np.transpose(image, (1, 2, 0))

    print(" - creating full size mask from tiles predictions")

    # Create mask placeholder
    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

    # Get prediction_paths for the image
    image_prediction_paths = [prediction_path for prediction_path in predictions_paths if inference_image_name in prediction_path]

    # Create full-size mask
    for image_prediction_path in image_prediction_paths:
        prediction_mask = cv2.imread(image_prediction_path, cv2.IMREAD_GRAYSCALE)

        image_prediction_name = pathlib.Path(image_prediction_path).stem
        x = int(image_prediction_name.split('_')[1])
        y = int(image_prediction_name.split('_')[2])

        prediction_mask = cv2.resize(prediction_mask, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        mask[y: y + tile_size, x: x + tile_size] = prediction_mask

        del prediction_mask

    # # Check if cortex mask and if yes use it to remove false positives outside of the cortex mask
    # anatomical_structure_path = os.path.join(inference_images_dir, inference_image_name + anatomical_structure_suffix)
    # if os.path.exists(anatomical_structure_path):
    #
    #    print(" - creating cortex mask")
    #    cortex_mask = np.zeros(shape=mask.shape, dtype=np.uint8)
    #    df_anatomical_structure = pd.read_json(anatomical_structure_path)
    #    for _, row in df_anatomical_structure.iterrows():
    #
    #        if row["properties"]["classification"]["name"] != "Cortex":
    #            continue
    #
    #        coordinates = row["geometry"]["coordinates"]
    #        if row["geometry"]["type"] == "MultiPolygon":
    #
    #            for polygon_pts in coordinates:
    #                cv2.fillPoly(cortex_mask, pts=[np.array(polygon_pts[0], dtype=np.int32)], color=1)
    #
    #        else:
    #            cv2.fillPoly(cortex_mask, pts=[np.array(coordinates[0], np.int32)], color=1)
    #
    #     print(" - removing glomeruli predictions outside of cortex mask")
    #     mask = cortex_mask * mask
    #
    #     del cortex_mask

    print(" - encoding mask to RLE")
    # Encode mask to RLE
    encoding = mask_to_rle(mask)
    encodings.append(encoding)
    ids.append(inference_image_name)

    overlays = list()
    colors = list()
    # Calculate metrics and create ground truth overlay
    if test:

        print(" - calculating scores")
        gt_mask = rle_to_mask(rle_encodings_gt.loc[inference_image_name]["encoding"], (image.shape[0], image.shape[1]))

        dice_scores.append(dice(mask, gt_mask))
        precision_scores.append(precision(mask, gt_mask))
        recall_scores.append(recall(mask, gt_mask))

        small_gt_mask = cv2.resize(gt_mask, (5000, 5000), interpolation=cv2.INTER_NEAREST)
        small_gt_mask = small_gt_mask * 255

        overlays.append(small_gt_mask)
        colors.append(1)

    print(" - saving downscaled overlay image")
    # Resize image and mask for visualization
    small_image = cv2.resize(image, (5000, 5000), interpolation=cv2.INTER_LINEAR)
    small_image = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)

    small_mask = cv2.resize(mask, (5000, 5000), interpolation=cv2.INTER_NEAREST)
    small_mask = small_mask * 255

    overlays.append(small_mask)
    colors.append(2)

    output_path = os.path.join(results_dir, inference_image_name + ".png")
    overlay_plot(small_image, overlays, colors, outputs_path=output_path)

    del image
    del mask
    del small_image
    del small_mask
    gc.collect()

print("\nSaving submission file")
# Populate pandas dataframe with encodings
df = pd.DataFrame()
df["id"] = ids
df["predicted"] = encodings
df.to_csv(submission_file_path, index=False)

print("\nSaving scores file")
if test:
    # Populate pandas dataframe with scores
    df = pd.DataFrame()
    df["id"] = ids
    df["dice"] = dice_scores
    df["precision"] = precision_scores
    df["recall"] = recall_scores
    df.to_csv(scores_file_path, index=False)
