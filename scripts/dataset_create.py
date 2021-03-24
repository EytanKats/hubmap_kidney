import os
import cv2
import glob
import pathlib
import numpy as np
import tifffile as tiff

from simple_converge.utils.dataset_utils import load_dataset_file, rle_to_mask, create_dataset

# Define settings
raw_images_template = "/data/eytank/datasets/hubmap_kidney/raw_data/images/train/*.tiff"

output_images_dir = "/data/eytank/datasets/hubmap_kidney/images_256"
output_masks_dir = "/data/eytank/datasets/hubmap_kidney/masks_256"
output_dataset_file_path = "/data/eytank/datasets/hubmap_kidney/ds_tile1024_step1024.csv"

train_dataset = True
rle_encodings_path = "/data/eytank/datasets/hubmap_kidney/raw_data/annotations/train/train.csv"

tile_size = 1024
tile_step = 128
resize_shape = (256, 256)
sat_thr = 40
num_pixels_thr = 10000

# Create output directories
if not os.path.exists(output_images_dir):
    os.mkdir(output_images_dir)

if not os.path.exists(output_masks_dir):
    os.mkdir(output_masks_dir)

# Load annotations
if train_dataset:
    df_rle_encodings = load_dataset_file(rle_encodings_path).set_index("id")

# Extract patches from raw images
raw_images_paths = glob.glob(raw_images_template)
for raw_image_path in raw_images_paths:

    # Read image
    raw_image_name = pathlib.Path(raw_image_path).stem
    print("Loading image {0}".format(raw_image_name))

    raw_image = tiff.imread(raw_image_path)
    if len(raw_image.shape) == 5:
        raw_image = raw_image.squeeze()
        raw_image = np.transpose(raw_image, (1, 2, 0))

    print("Loaded image with shape ({0}, {1})".format(raw_image.shape[0], raw_image.shape[1]))

    # Get binary mask from RLE encoding
    if train_dataset:
        print("Creating mask from RLE encodings")
        mask = rle_to_mask(df_rle_encodings.loc[raw_image_name]["encoding"], (raw_image.shape[0], raw_image.shape[1]))

    # Crop tiles that have sufficient saturation from the raw image
    print("Cropping tiles from the image")
    num_valid_tiles = 0
    num_rejected_by_saturation_tiles = 0

    if train_dataset:
        num_valid_tiles_without_glomeruli = 0
        num_rejected_by_saturation_tiles_with_glomeruli = 0

    for x in range(0, raw_image.shape[1] - tile_size, tile_step):
        for y in range(0, raw_image.shape[0] - tile_size, tile_step):

            # Crop tile
            tile_image = raw_image[y: y + tile_size, x: x + tile_size]

            if train_dataset:
                tile_mask = mask[y: y + tile_size, x: x + tile_size]
                tile_mask = (tile_mask * 255).astype(np.uint8)

                num_mask_positive_pixels = np.sum(tile_mask)

            # Check saturation
            hsv_tile_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_tile_image)

            if (s > sat_thr).sum() > num_pixels_thr:  # Save tile

                tile_image = cv2.resize(tile_image, resize_shape, interpolation=cv2.INTER_LINEAR)

                output_path_image = os.path.join(output_images_dir, raw_image_name + "_" + str(x) + "_" + str(y) + "_" + "image" + ".png")
                cv2.imwrite(output_path_image, tile_image)

                num_valid_tiles += 1

                if train_dataset:

                    tile_mask = cv2.resize(tile_mask, resize_shape, interpolation=cv2.INTER_NEAREST)

                    output_path_mask = os.path.join(output_masks_dir, raw_image_name + "_" + str(x) + "_" + str(y) + "_" + "mask" + ".png")
                    cv2.imwrite(output_path_mask, tile_mask)

                    if num_mask_positive_pixels == 0:
                        num_valid_tiles_without_glomeruli += 1

            else:  # Reject tile

                num_rejected_by_saturation_tiles += 1

                if train_dataset and num_mask_positive_pixels > 0:
                    num_rejected_by_saturation_tiles_with_glomeruli += 1

                continue

    print("Number of valid tiles: {0}".format(num_valid_tiles))
    print("Number of tiles rejected by saturation: {0}".format(num_rejected_by_saturation_tiles))

    if train_dataset:
        print("Number of valid tiles without glomeruli: {0}".format(num_valid_tiles_without_glomeruli))
        print("Number of tiles rejected by saturation with glomeruli: {0}".format(num_rejected_by_saturation_tiles_with_glomeruli))

if train_dataset:  # Create dataset file
    print("Creating dataset file")
    df = create_dataset(data_template=output_images_dir + "/*.png",
                        mask_template=output_masks_dir + "/*.png",
                        save_dataset_file=False,
                        output_dataset_file_path=output_dataset_file_path)

    # Dataset file will contain only images without overlap
    image_basename_split = df["image_basename"].apply(lambda el: el.split('_'))
    df["raw_image_id"] = [el[0] for el in image_basename_split]
    df["x"] = [int(el[1]) for el in image_basename_split]
    df["y"] = [int(el[2]) for el in image_basename_split]

    df = df[(df["x"] % tile_size == 0) & (df["y"] % tile_size == 0)]

    df.to_csv(output_dataset_file_path, index=False)

print("End of Script")
