import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from utils.RunMode import RunMode
from utils.dataset_utils import mask_to_rle
from plots.plots import contours_plot
from metrics.metrics import dice, recall, precision
from dataset.BaseDataset import BaseDataset


class Dataset(BaseDataset):

    """
    This class expands BaseDataset and adds specific functionality for glomeruli segmentation
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(Dataset, self).__init__()

        # Fields to be filled by parsing
        self.mask_path_column = ""

        self.resize_shape = (256, 256)

        self.postprocessing_thr = 0.5
        self.remove_small_cc = True

        self.predictions_dir = "predictions"

        # Fields to be filled during execution
        self.metric_scores = pd.DataFrame([], columns=["dice", "precision", "recall"])

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Dataset, self).parse_args(**kwargs)

        if "mask_path_column" in self.params.keys():
            self.mask_path_column = self.params["mask_path_column"]

        if "resize_shape" in self.params.keys():
            self.resize_shape = self.params["resize_shape"]

        if "postprocessing_thr" in self.params.keys():
            self.postprocessing_thr = self.params["postprocessing_thr"]

        if "remove_small_cc" in self.params.keys():
            self.remove_small_cc = self.params["remove_small_cc"]

        if "predictions_dir" in self.params.keys():
            self.predictions_dir = self.params["predictions_dir"]

    def initialize_dataset(self, run_mode=RunMode.TRAINING):

        super(Dataset, self).initialize_dataset(run_mode=run_mode)

        # Set display options for pandas
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

    def _get_data(self, info_row):

        data = cv2.imread(info_row[self.data_path_column])
        return data

    def _get_label(self, info_row):

        mask = cv2.imread(info_row[self.mask_path_column], cv2.IMREAD_GRAYSCALE)
        return mask

    def _apply_augmentations(self, data, label):

        return data, label

    def _apply_preprocessing(self, data, label, info_row, run_mode=RunMode.TRAINING):

        # Data preprocessing
        data = cv2.resize(data, self.resize_shape, interpolation=cv2.INTER_LINEAR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        data = data.astype(np.float32)
        data = tf.keras.applications.resnet50.preprocess_input(data)

        # Label preprocessing
        label = cv2.resize(label, self.resize_shape, interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, axis=2)

        label = label.astype('float32')
        label = label / 255.0  # normalize to the range 0-1

        return data, label

    def apply_postprocessing(self, test_predictions, test_data, original_test_data, data_info, fold_num, run_mode=RunMode.TRAINING):

        post_processed_predictions = list()
        for prediction_idx, prediction in enumerate(test_predictions):

            # Apply threshold on predicted mask
            _, post_processed_prediction = cv2.threshold(prediction[..., 0], self.postprocessing_thr, 1, cv2.THRESH_BINARY)

            # Fill connected components
            post_processed_prediction = self.remove_holes(post_processed_prediction)

            post_processed_predictions.append(post_processed_prediction)

        return post_processed_predictions

    def remove_holes(self, mask):

        postprocessed_mask = np.copy(mask)
        im_floodfill = np.copy(postprocessed_mask)

        h, w = postprocessed_mask.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0, 0), 2)
        cv2.floodFill(im_floodfill, mask, (0, self.resize_shape[0] - 1), 2)
        cv2.floodFill(im_floodfill, mask, (self.resize_shape[1] - 1, 0), 2)
        cv2.floodFill(im_floodfill, mask, (self.resize_shape[1] - 1, self.resize_shape[0] - 1), 2)

        postprocessed_mask[im_floodfill != 2] = 1

        return postprocessed_mask

    def calculate_fold_metrics(self, test_predictions, test_data, *_):

        labels = test_data[1]

        # Calculate metrics
        dice_scores = [dice(test_predictions[idx], labels[idx]) for idx in range(len(test_predictions))]
        recall_scores = [recall(test_predictions[idx], labels[idx]) for idx in range(len(test_predictions))]
        precision_scores = [precision(test_predictions[idx], labels[idx]) for idx in range(len(test_predictions))]

        # Save metrics to calculate statistics over folds
        stack = np.hstack((dice_scores, recall_scores, precision_scores))
        df = pd.DataFrame(stack, columns=["dice", "precision", "recall"])
        self.metrics_scores = self.metrics_scores.append(df)

        self._log(df.describe())

    def log_metrics(self, output_folder):

        self._log(self.metrics_scores.describe())
        self.metrics_scores.to_csv(os.path.join(output_folder, "metrics.csv"), index=False)

        sns.set(style="whitegrid")
        sns.boxplot(data=self.metrics_scores)
        plt.savefig(os.path.join(output_folder, "boxplot.png"))

    def _save_data(self, predictions, data, original_data, data_info, output_dir, run_mode):

        # Create folder for auxiliary outputs
        auxiliary_output_dir = os.path.join(output_dir, "auxiliary")
        if not os.path.exists(auxiliary_output_dir):
            os.makedirs(auxiliary_output_dir)

        for prediction_idx, prediction in enumerate(predictions[0]):

            info_row = data_info.iloc[prediction_idx]
            img_name = os.path.basename(info_row[self.data_path_column])

            image = original_data[0][prediction_idx]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = list()
            contour_colors = list()

            resized_prediction = cv2.resize(prediction[..., 0], (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            resized_prediction = resized_prediction.astype(np.uint8) * 255
            masks.append(resized_prediction)
            contour_colors.append((0, 0, 255))  # blue contour for prediction

            if run_mode == RunMode.TEST:
                masks.append(original_data[0][prediction_idx])
                contour_colors.append((0, 255, 0))  # green contour for ground truth

            output_path = os.path.join(output_dir, "auxiliary", img_name)
            contours_plot(image, masks, contour_colors, outputs_path=output_path)

            auxiliary_output_path = os.path.join(output_dir, "auxiliary", img_name)
            cv2.imwrite(auxiliary_output_path, resized_prediction)

    def save_tested_data(self, test_predictions, test_data, original_test_data, fold_test_info, fold_num, output_folder):

        output_folder = os.path.join(output_folder, self.predictions_dir)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self._save_data(test_predictions, test_data, original_test_data, fold_test_info, output_folder, RunMode.TEST)

    def save_inferenced_data(self, inference_predictions, inference_data, original_inference_data, batch_inference_info, batch_num, output_folder):

        output_folder = os.path.join(output_folder, self.predictions_dir)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self._save_data(inference_predictions, inference_data, original_inference_data, batch_inference_info, output_folder, RunMode.INFERENCE)

