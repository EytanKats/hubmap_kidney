import os
import shutil
from simple_converge.utils.dataset_utils import load_dataset_file


simulation_dir = "C:/Users/212761522/Box Sync/Work/EduProjects/HuBMAP_Kidney/Sims/2021.03.27_smefficientnetb4_updateddata"
dst_dir = "C:/Users/212761522/Box Sync/Work/EduProjects/HuBMAP_Kidney/Sims/2021.03.27_smefficientnetb4_updateddata/predictions_with_dice_lower_than_0.8"
results_file_path = "C:/Users/212761522/Box Sync/Work/EduProjects/HuBMAP_Kidney/Sims/2021.03.27_smefficientnetb4_updateddata/metrics.csv"
dice_thr = 0.8

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

results_df = load_dataset_file(results_file_path)
for _, row in results_df.iterrows():

    if row["dice"] < dice_thr:
        src_path = os.path.join(simulation_dir, str(int(row["fold"])), row["image_basename"])
        dst_path = os.path.join(dst_dir, row["image_basename"])

        shutil.copy(src_path, dst_path)
