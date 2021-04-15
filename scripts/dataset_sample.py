import cv2
import pandas as pd
from simple_converge.utils.dataset_utils import load_dataset_file

# Calculate number of glomeruli pixels for every tile
num_positive_pixels = list()
df = load_dataset_file("/data/eytank/datasets/hubmap_kidney/ds_tile1024_step1024.csv")
for _, row in df.iterrows():
    mask_path = row["mask"]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    num_positive_pixels.append((mask == 255).sum())

# Divide dataset to tiles with at least ~5% of the pixels are positives and tiles with less than ~5% of the pixels are positive
df["positive_pixels"] = num_positive_pixels
neg_df = df[df["positive_pixels"] < 3000]
pos_df = df[df["positive_pixels"] >= 3000]

print("Number of tiles in dataset is {0}".format(df.shape[0]))
print("Number of tiles with at least ~5% of the positive pixels is {0}".format(pos_df.shape[0]))
print("Number of tiles with less than ~5% of the positive pixels is {0}".format(neg_df.shape[0]))

# Equalize number of positive and negative tiles in dataset
neg_df = neg_df.sample(pos_df.shape[0])
sampled_df = pd.concat([pos_df, neg_df])

print("Number of tiles in dataset with equal number of positive and negative tiles is {0}".format(sampled_df.shape[0]))

sampled_df.to_csv("/data/eytank/datasets/hubmap_kidney/ds_tile1024_step1024_sampled.csv", index=False)

print("End of script")
