import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as alb

import seaborn as sns
import matplotlib.pyplot as plt

from simple_converge.utils.RunMode import RunMode
from simple_converge.plots.plots import contours_plot
from simple_converge.metrics.metrics import dice, recall, precision
from simple_converge.data.BaseDataset import BaseDataset


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
        self.data_path_column = ""
        self.mask_path_column = ""

        self.resize_shape = (256, 256)

        self.postprocessing_thr = 0.5
        self.remove_small_cc = True

        self.predictions_dir = "predictions"

        # Fields to be filled during execution
        self.metrics_scores = pd.DataFrame([], columns=["dice", "precision", "recall"])

        self.x_offset = 0  # offset to augment file name during training
        self.y_offset = 0  # offset to augment file name during training

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Dataset, self).parse_args(**kwargs)

        if "data_path_column" in self.params.keys():
            self.data_path_column = self.params["data_path_column"]

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

        # Set display options for pandas
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

    def get_data(self, info_row, run_mode=RunMode.TRAINING):

        if run_mode != run_mode.TRAINING:
            data = cv2.imread(info_row[self.data_path_column])

        else:

            # Split anchor data name to components: name, x, y, suffix
            anchor_data_path = info_row[self.data_path_column]
            anchor_base_dir = os.path.dirname(anchor_data_path)
            anchor_data_name = os.path.basename(anchor_data_path)
            anchor_data_name_split = anchor_data_name.split("_")
            anchor_name = anchor_data_name_split[0]
            anchor_x = int(anchor_data_name_split[1])
            anchor_y = int(anchor_data_name_split[2])
            anchor_suffix = anchor_data_name_split[3]

            # Augment x and y coordinates
            self.x_offset = np.random.randint(0, 8) * 128
            self.y_offset = np.random.randint(0, 8) * 128
            augmented_x = anchor_x + self.x_offset
            augmented_y = anchor_y + self.y_offset

            # Combine augmented path
            augmented_anchor_data_name = anchor_name + "_" + str(augmented_x) + "_" + str(augmented_y) + "_" + anchor_suffix
            augmented_anchor_data_path = os.path.join(anchor_base_dir, augmented_anchor_data_name)

            # Check that data with augmented name exists and if not try another time
            cnt = 0
            while not os.path.exists(augmented_anchor_data_path) and cnt < 8:

                self.x_offset = np.random.randint(0, 8) * 128
                self.y_offset = np.random.randint(0, 8) * 128
                augmented_x = anchor_x + self.x_offset
                augmented_y = anchor_x + self.y_offset

                augmented_anchor_data_name = anchor_name + "_" + str(augmented_x) + "_" + str(augmented_y) + "_" + anchor_suffix
                augmented_anchor_data_path = os.path.join(anchor_base_dir, augmented_anchor_data_name)
                cnt += 1

            if cnt < 8:  # data with augmented name exists
                data = cv2.imread(augmented_anchor_data_path)
            else:  # data with augmented name doesn't exists
                data = cv2.imread(anchor_data_path)

        return data

    def get_label(self, info_row, run_mode=RunMode.TRAINING):

        if run_mode != run_mode.TRAINING:
            mask = cv2.imread(info_row[self.mask_path_column], cv2.IMREAD_GRAYSCALE)

        else:

            # Get augmented label path
            anchor_label_path = info_row[self.mask_path_column]
            anchor_base_dir = os.path.dirname(anchor_label_path)
            anchor_label_name = os.path.basename(anchor_label_path)
            anchor_label_name_split = anchor_label_name.split("_")
            anchor_name = anchor_label_name_split[0]
            augmented_x = self.x_offset + int(anchor_label_name_split[1])
            augmented_y = self.y_offset + int(anchor_label_name_split[2])
            anchor_suffix = anchor_label_name_split[3]

            augmented_anchor_label_name = anchor_name + "_" + str(augmented_x) + "_" + str(augmented_y) + "_" + anchor_suffix
            augmented_anchor_label_path = os.path.join(anchor_base_dir, augmented_anchor_label_name)

            if os.path.exists(augmented_anchor_label_path):
                mask = cv2.imread(augmented_anchor_label_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = cv2.imread(anchor_label_path, cv2.IMREAD_GRAYSCALE)

        return mask

    def apply_augmentations(self, data, label=None,
                            info_row=None, run_mode=RunMode.TRAINING):

        transform = alb.Compose([alb.RandomRotate90(always_apply=True),
                                 alb.HorizontalFlip(p=0.5),
                                 alb.VerticalFlip(p=0.5)])

        augmented = transform(image=data, mask=label)
        data_aug = augmented["image"]
        label_aug = augmented["mask"]

        # img_id = np.random.randint(low=0, high=30)
        # cv2.imwrite("/data/eytank/simulations/hubmap_kidney/img_" + str(img_id) + ".png", data)
        # cv2.imwrite("/data/eytank/simulations/hubmap_kidney/img_aug_" + str(img_id) + ".png", data_aug)
        # cv2.imwrite("/data/eytank/simulations/hubmap_kidney/lbl_" + str(img_id) + ".png", label)
        # cv2.imwrite("/data/eytank/simulations/hubmap_kidney/lbl_aug_" + str(img_id) + ".png", label_aug)

        return data_aug, label_aug

    def apply_preprocessing(self, data, label=None,
                            info_row=None, run_mode=RunMode.TRAINING):

        # Data preprocessing
        data = cv2.resize(data, self.resize_shape, interpolation=cv2.INTER_LINEAR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        data = data.astype(np.float32)
        data = tf.keras.applications.resnet50.preprocess_input(data)

        # Label preprocessing
        if label is not None:
            label = cv2.resize(label, self.resize_shape, interpolation=cv2.INTER_NEAREST)
            label = np.expand_dims(label, axis=2)

            label = label.astype('float32')
            label = label / 255.0  # normalize to the range 0-1

        return data, label

    def apply_postprocessing_on_predictions_batch(self,
                                                  predictions,
                                                  preprocessed_data_and_labels=None,
                                                  not_preprocessed_data_and_labels=None,
                                                  batch_df=None,
                                                  batch_id=0,
                                                  run_mode=RunMode.TEST):

        post_processed_predictions = list()
        for prediction_idx, prediction in enumerate(predictions):

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

    def calculate_batch_metrics(self,
                                postprocessed_predictions,
                                preprocessed_data_and_labels=None,
                                not_preprocessed_data_and_labels=None,
                                batch_df=None,
                                batch_id=0,
                                output_dir=None):

        labels = preprocessed_data_and_labels[1]

        # Calculate metrics
        dice_scores = [[dice(postprocessed_predictions[idx], labels[idx][..., 0])] for idx in range(len(postprocessed_predictions))]
        recall_scores = [[recall(postprocessed_predictions[idx], labels[idx][..., 0])] for idx in range(len(postprocessed_predictions))]
        precision_scores = [[precision(postprocessed_predictions[idx], labels[idx][..., 0])] for idx in range(len(postprocessed_predictions))]

        # Save metrics to calculate statistics over folds
        stack = np.hstack((dice_scores, recall_scores, precision_scores))
        df = pd.DataFrame(stack, columns=["dice", "recall", "precision"])
        df["image_basename"] = batch_df["image_basename"]
        df["fold"] = batch_id
        self.metrics_scores = self.metrics_scores.append(df)

        self.logger.log(df.describe())

    def aggregate_metrics_for_all_batches(self,
                                          output_dir=None):

        self.logger.log(self.metrics_scores.describe())
        self.metrics_scores.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

        sns.set(style="whitegrid")
        sns.boxplot(data=self.metrics_scores)
        plt.savefig(os.path.join(output_dir, "boxplot.png"))

    def save_data_batch(self,
                        postprocessed_predictions,
                        output_dir,
                        not_postprocessed_predictions=None,
                        preprocessed_data_and_labels=None,
                        not_preprocessed_data_and_labels=None,
                        batch_df=None,
                        batch_id=0):

        output_dir = os.path.join(output_dir, self.predictions_dir)

        # Create folder for auxiliary outputs
        auxiliary_output_dir = os.path.join(output_dir, "auxiliary")
        if not os.path.exists(auxiliary_output_dir):
            os.makedirs(auxiliary_output_dir)

        for prediction_idx, prediction in enumerate(postprocessed_predictions):

            info_row = batch_df.iloc[prediction_idx]
            img_name = os.path.basename(info_row[self.data_path_column])

            image = not_preprocessed_data_and_labels[0][prediction_idx]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = list()
            contour_colors = list()

            not_postprocessed_prediction = not_postprocessed_predictions[prediction_idx][..., 0]
            resized_not_postprocessed_prediction = cv2.resize(not_postprocessed_prediction, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            resized_not_postprocessed_prediction = (resized_not_postprocessed_prediction * 255).astype(np.uint8)

            resized_prediction = cv2.resize(prediction, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            resized_prediction = resized_prediction.astype(np.uint8) * 255
            masks.append(resized_prediction)
            contour_colors.append((0, 0, 255))  # blue contour for prediction

            masks.append(not_preprocessed_data_and_labels[1][prediction_idx])
            contour_colors.append((0, 255, 0))  # green contour for ground truth

            output_path = os.path.join(output_dir, img_name)
            contours_plot(image, masks, contour_colors, outputs_path=output_path)

            auxiliary_output_path = os.path.join(output_dir, "auxiliary", img_name)
            cv2.imwrite(auxiliary_output_path, resized_not_postprocessed_prediction)

    # def save_tested_data(self, test_predictions, test_data, original_test_data, fold_test_info, fold_num, output_folder):
    #
    #     output_folder = os.path.join(output_folder, self.predictions_dir)
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #
    #     self._save_data(test_predictions, test_data, original_test_data, fold_test_info, output_folder, RunMode.TEST)
    #
    # def save_inferenced_data(self, inference_predictions, inference_data, original_inference_data, batch_inference_info, batch_num, output_folder):
    #
    #     output_folder = os.path.join(output_folder, self.predictions_dir)
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #
    #     self._save_data(inference_predictions, inference_data, original_inference_data, batch_inference_info, output_folder, RunMode.INFERENCE)

