import os
import cv2
import glob
import pathlib
import numpy as np
import tifffile as tiff


inference_images_template = "../../Datasets/HuBMAP_Kidney/raw_data/images/test/*.tiff"
tiles_dir = "../../Datasets/HuBMAP_Kidney/test"

tile_size = 1000
tile_step = 1000
resize_shape = (256, 256)
sat_thr = 40
num_pixels_thr = 10000

# Create inference dataset
inference_images_paths = glob.glob(inference_images_template)
for inference_image_path in inference_images_paths:

    # Read image
    image_name = pathlib.Path(inference_image_path).stem
    image = tiff.imread(inference_image_path)

    if len(image.shape) == 5:
        image = image.squeeze()
        image = np.transpose(image, (1, 2, 0))

    print(image.shape)
    for x in range(0, image.shape[1] - tile_size, tile_step):
        for y in range(0, image.shape[0] - tile_size, tile_step):

            tile_image = image[y: y + tile_size, x: x + tile_size]
            hsv_tile_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_tile_image)
            if (s > sat_thr).sum() > num_pixels_thr:

                output_path_image = os.path.join(tiles_dir, image_name + "_" + str(x) + "_" + str(y) + "_" + ".png")
                cv2.resize(tile_image, resize_shape, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(output_path_image, tile_image)

print("End of script")
