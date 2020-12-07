import os
import cv2
import numpy as np
import tifffile as tiff

from utils.dataset_utils import load_dataset_file, rle_to_mask, create_dataset

raw_images_dir = "../../Datasets/HuBMAP_Kidney/raw_data/images/train"
rle_encodings_path = "../../Datasets/HuBMAP_Kidney/raw_data/annotations/train/train.csv"

output_images_dir = "../../Datasets/HuBMAP_Kidney/ds_1/images"
output_masks_dir = "../../Datasets/HuBMAP_Kidney/ds_1/masks"
output_dataset_file_path = "../../Datasets/HuBMAP_Kidney/ds_1/ds_1.csv"

tile_size = 1000
tile_step = 1000

df_rle_encodings = load_dataset_file(rle_encodings_path).set_index("id")
for idx, rle_encoding in df_rle_encodings.iterrows():

    # Read image
    image_path = os.path.join(raw_images_dir, idx + ".tiff")
    image = tiff.imread(image_path)

    if len(image.shape) == 5:
        image = image.squeeze()
        image = np.transpose(image, (1, 2, 0))

    # Get binary mask from RLE encoding
    mask = rle_to_mask(rle_encoding["encoding"], (image.shape[0], image.shape[1]))

    print(image.shape)
    for x in range(0, image.shape[1] - tile_size, tile_step):
        for y in range(0, image.shape[0] - tile_size, tile_step):

            tile_mask = mask[y: y + tile_size, x: x + tile_size]
            num_mask_positive_pixels = np.sum(tile_mask)

            if num_mask_positive_pixels > 100:

                output_path_mask = os.path.join(output_images_dir, idx + "_" + str(x) + "_" + str(y) + "_" + "mask" + ".png")
                tile_mask = (tile_mask * 255).astype(np.uint8)
                cv2.imwrite(output_path_mask, tile_mask)

                output_path_image = os.path.join(output_masks_dir, idx + "_" + str(x) + "_" + str(y) + "_" + "image" + ".png")
                tile_image = image[y: y + tile_size, x: x + tile_size]
                cv2.imwrite(output_path_image, tile_image)

df = create_dataset(data_template=output_images_dir + "/*.png",
                    mask_template=output_masks_dir + "/*.png",
                    save_dataset_file=False,
                    output_dataset_file_path=output_dataset_file_path)

df["raw_image_id"] = df["image_basename"].apply(lambda x: x.split('_')[0])
df.to_csv(output_dataset_file_path, index=False)

print("End of Script")