"""
Data Generator for train and validation datasets
"""

import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from src.augmentations import parent_aug
from src.keypoints import KeypointsNonFlatIndex

logger = logging.getLogger(__name__)


class GISLRSequence(tf.keras.utils.Sequence):
    def __init__(
            self, df: pd.DataFrame,
            x_col: str,
            y_col: str,
            sample_size: int,
            batch_size: int,
            shuffle: bool = True,
            example_set: str = "train"
        ) -> None:
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.n_classes = df[y_col].nunique()
        self.example_set = example_set

        self.x_files_path = df[x_col]
        self.y = df[y_col]

        self.indices = df.index.to_list()

    def __len__(self):
        return (len(self.x_files_path) // self.batch_size)

    def __getitem__(self, idx):
        subset = self.indices[(idx * self.batch_size):((idx + 1) * self.batch_size)]
        batch_x = self.x_files_path[subset]
        batch_y = self.y[subset]

        X, y = self._get_data(batch_x, batch_y)

        return X, tf.one_hot(y, depth=self.n_classes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _get_data(self, batch_x, batch_y):
        temp_X, temp_y = [], []

        for idx, idy in zip(batch_x, batch_y):
            temp = self._get_transformed_data(idx)

            temp_X.append(temp)
            temp_y.append(idy)

        return np.array(temp_X), np.array(temp_y)

    def _get_transformed_data(self, dataset_file_path) -> np.ndarray:
        all_frames = np.load(dataset_file_path)
        subset_imgs = np.linspace(0, all_frames.shape[0], self.sample_size, dtype=np.int8, endpoint=False)
        all_frames = all_frames[subset_imgs, :]

        # Subsetting x,y coords
        # all_frames = all_frames[:, :543*2]

        logger.info(f"Found: {all_frames.shape} frames in {dataset_file_path}")

        all_frames = np.where(np.isnan(all_frames), 0, all_frames)
        all_frames = np.resize(all_frames, (all_frames.shape[0], 543, 3))

        subset_pts = list(KeypointsNonFlatIndex.lips) + list(KeypointsNonFlatIndex.eyes)\
            + list(KeypointsNonFlatIndex.pose) + list(KeypointsNonFlatIndex.left_hand)\
            + list(KeypointsNonFlatIndex.right_hand)

        if self.example_set == "train":
            # Augmentation args
            aug_kwargs = dict(
                # p_hflip=0.3,
                p_rotation=0.3,
                p_scaling=0.3
            )

            all_frames = parent_aug(
                all_frames, **aug_kwargs
            )

        all_frames = all_frames[:, subset_pts, :1]
        return np.resize(all_frames, (all_frames.shape[0], all_frames.shape[1] * 2))
