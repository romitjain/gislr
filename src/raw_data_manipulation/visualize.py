"""
Given a numpy file, visualize the keypoints as a video
"""

import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
import logging
from mediapipe.framework.formats import landmark_pb2

import src.augmentations as augmentations
from src.keypoints import index_labels, TOTAL_PTS, KeypointsNonFlatIndex, connections_type

logger = logging.getLogger(__name__)


def get_pts_to_img(data: np.ndarray, img_size: tuple = (512, 512, 3)) -> np.ndarray:
    """
    Convert x, y, z coordinates to an image
    """
    image = np.zeros(img_size, np.uint8)

    pts = {
        "face": [],
        "left_hand": [],
        "right_hand": [],
        "pose": []
    }

    for body_part in pts.keys():
        # Get x, y, z coordinates from flattened npy file
        temp = data[index_labels[body_part][0]: index_labels[body_part][1], :]

        logger.info("body part", body_part, temp.shape)

        for elem in temp:
            pts[body_part].append(
                landmark_pb2.NormalizedLandmark(
                    x=elem[0],
                    y=elem[1],
                    z=elem[2],
                    visibility=1.0
                )
            )

    for body_part, landmarks in pts.items():
        landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark = landmarks)

        mp.solutions.drawing_utils.draw_landmarks(
            image,
            landmark_subset,
            connections_type[body_part]
        )

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--keypoints',
        type=str,
        required=True
    )

    args = parser.parse_args()

    all_frames = np.load(os.path.abspath(args.keypoints))

    all_frames = np.where(np.isnan(all_frames), 0, all_frames)
    all_frames = np.resize(all_frames, (all_frames.shape[0], TOTAL_PTS, 3))

    print(f"Loaded numpy with shape: {all_frames.shape}")

    subset_pts = list(KeypointsNonFlatIndex.left_hand) + \
        list(KeypointsNonFlatIndex.right_hand) + \
        list(KeypointsNonFlatIndex.eyes) + \
        list(KeypointsNonFlatIndex.lips) + \
        list(KeypointsNonFlatIndex.pose)

    # Augmentation args
    aug_kwargs = dict(
        p_hflip=1.0,
        p_rotation=1.0,
        p_scaling=1.0
    )

    all_frames = augmentations.parent_aug(
        all_frames, **aug_kwargs
    )

    all_frames = all_frames[:, sorted(set(subset_pts)), :]

    for idx, single_frame in enumerate(all_frames):
        print(f"Loading frame: {idx}")
        image = get_pts_to_img(single_frame)

        cv2.imshow('MediaPipe Holistic', image)

        key = cv2.waitKey(1000)
        if key == ord('q'):
            break

        cv2.waitKey(-1)

    cv2.destroyAllWindows()
