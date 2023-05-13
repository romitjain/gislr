"""Converts the raw data provided to numpy files"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import get_all_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_dataset_path',
        type=str,
        required=True,
        help="Root path of kaggle dataset"
    )

    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help="Relative path to store the generated dataset"
    )

    args = parser.parse_args()

    train_csv = os.path.abspath(os.path.join(args.train_dataset_path, "train.csv"))
    train_landmark_files = os.path.abspath(os.path.join(args.train_dataset_path, "train_landmark_files"))

    train = pd.read_csv(train_csv)
    all_files = get_all_files(train_landmark_files)

    for fl in tqdm(all_files):
        df = pd.read_parquet(fl)
        qpath = fl.split(os.path.abspath(args.train_dataset_path))[1][1:]

        sequence_id = train[train.path == qpath].sequence_id.iloc[0]

        all_imgs = np.array(
            df.loc[:, ["x", "y", "z"]],
            dtype=np.float16
        )
        all_imgs = all_imgs.reshape((df.frame.nunique(), 543 * 3), order="F")

        with open(
            os.path.join(os.path.abspath(args.save_path), str(sequence_id) + ".npy"), "wb"
        ) as fr:
            np.save(file=fr, arr=all_imgs)
