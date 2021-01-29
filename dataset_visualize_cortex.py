import os
import cv2
import numpy as np
import tifffile as tiff

from plots.plots import overlay_plot
from utils.dataset_utils import load_dataset_file, rle_to_mask


raw_images_dir = "../../Datasets/HuBMAP_Kidney/raw_data/images/train"
rle_encodings_path = "../../Datasets/HuBMAP_Kidney/raw_data/annotations/train/train.csv"
anatomical_structure_dir = "../../Datasets/HuBMAP_Kidney/raw_data/annotations/train"
anatomical_structure_suffix = "-anatomical-structure.json"
output_dir = "../../Datasets/HuBMAP_Kidney/visualization"

df_rle_encodings = load_dataset_file(rle_encodings_path).set_index("id")
for idx, rle_encoding in df_rle_encodings.iterrows():

    # Read image
    image_path = os.path.join(raw_images_dir, idx + ".tiff")
    image = tiff.imread(image_path)

    if len(image.shape) == 5:
        image = image.squeeze()
        image = np.transpose(image, (1, 2, 0))

    # Get binary glomeruli mask from RLE encoding
    glomeruli_mask = rle_to_mask(rle_encoding["encoding"], (image.shape[0], image.shape[1]))

    # Get binary cortex mask
    cortex_mask = np.zeros(shape=glomeruli_mask.shape, dtype=np.uint8)
    anatomical_structure_path = os.path.join(anatomical_structure_dir, idx + anatomical_structure_suffix)
    df_anatomical_structure = load_dataset_file(anatomical_structure_path)
    for _, row in df_anatomical_structure.iterrows():

        if row["properties"]["classification"]["name"] != "Cortex":
            continue

        coordinates = row["geometry"]["coordinates"]
        if row["geometry"]["type"] == "MultiPolygon":

            for polygon_pts in coordinates:
                cv2.fillPoly(cortex_mask, pts=[np.array(polygon_pts[0], dtype=np.int32)], color=1)

        else:
            cv2.fillPoly(cortex_mask, pts=[np.array(coordinates[0], np.int32)], color=1)

    # Resize image and mask for visualization
    small_image = cv2.resize(image, (5000, 5000), interpolation=cv2.INTER_LINEAR)
    small_image = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)

    small_glomeruli_mask = cv2.resize(glomeruli_mask, (5000, 5000), interpolation=cv2.INTER_NEAREST)
    small_cortex_mask = cv2.resize(cortex_mask, (5000, 5000), interpolation=cv2.INTER_NEAREST)

    overlays = [small_glomeruli_mask * 255, small_cortex_mask * 255]
    colors = [2, 1]

    output_path = os.path.join(output_dir, idx + ".png")
    overlay_plot(small_image, overlays, colors, outputs_path=output_path)

    print("Image {0} processed".format(idx))

print("End of script")
