"""
Augmentation for keypoints

# How to counter images that does not have complete information
1. hflip for all frames
2. make one random hand as 0
3. Switch left and right hands

"""

import numpy as np
from logging import getLogger
import albumentations as A
from src.keypoints import KeypointsIndex, KeypointsNonFlatIndex

logger = getLogger(__name__)

def parent_aug(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Parent augmentation function that decides what augmentations to apply

    Args:
        data (np.ndarray): Original unmodified data

    Returns:
        np.ndarray: Augmented data
    """

    data = hflip(data, **kwargs)
    data = augment_np_keypoints(data, **kwargs)

    return data


def aug_ffill(data: np.ndarray, **kwargs) -> np.ndarray:
    og_size = data.shape[0]
    for i in range(0, kwargs.get("sample_size")-og_size):
        frame_to_add = i % og_size

        data = np.concatenate((
            data, np.expand_dims(data[frame_to_add], axis=0)
        ))

    return data


def aug_bfill(data: np.ndarray, **kwargs) -> np.ndarray:
    og_size = data.shape[0]
    for i in range(0, kwargs.get("sample_size")-og_size):
        frame_to_add = abs(og_size-i) % og_size

        data = np.concatenate((
            data, np.expand_dims(data[frame_to_add], axis=0)
        ))

    return data


def aug_hflip(data: np.ndarray, **kwargs) -> np.ndarray:
    x_pts_rh, y_pts_rh, z_pts_rh = data[:, KeypointsIndex.right_hand_x_index], data[:, KeypointsIndex.right_hand_y_index], data[:, KeypointsIndex.right_hand_z_index]
    x_pts_lh, y_pts_lh, z_pts_lh = data[:, KeypointsIndex.left_hand_x_index], data[:, KeypointsIndex.left_hand_y_index], data[:, KeypointsIndex.left_hand_z_index]

    data[:, KeypointsIndex.right_hand_x_index] = 1 - x_pts_lh
    data[:, KeypointsIndex.left_hand_x_index] = 1 - x_pts_rh

    data[:, KeypointsIndex.right_hand_y_index] = y_pts_lh
    data[:, KeypointsIndex.left_hand_y_index] = y_pts_rh

    data[:, KeypointsIndex.right_hand_z_index] = z_pts_lh
    data[:, KeypointsIndex.left_hand_z_index] = z_pts_rh

    return data


def hflip(all_frames: np.ndarray, **kwargs) -> np.ndarray:

    if np.random.uniform() > kwargs.get("p_hflip", 0):
        return all_frames

    data = all_frames.copy()

    # Switch hand index
    x_pts_rh, y_pts_rh, z_pts_rh = data[:, KeypointsNonFlatIndex.right_hand, 0], \
        data[:, KeypointsNonFlatIndex.right_hand, 1], \
        data[:, KeypointsNonFlatIndex.right_hand, 2]

    x_pts_lh, y_pts_lh, z_pts_lh = data[:, KeypointsNonFlatIndex.left_hand, 0], \
        data[:, KeypointsNonFlatIndex.left_hand, 1], \
        data[:, KeypointsNonFlatIndex.left_hand, 2]

    data[:, KeypointsNonFlatIndex.right_hand, 0] = 1 - x_pts_lh
    data[:, KeypointsNonFlatIndex.left_hand, 0] = 1 - x_pts_rh

    data[:, KeypointsNonFlatIndex.right_hand, 1] = y_pts_lh
    data[:, KeypointsNonFlatIndex.left_hand, 1] = y_pts_rh

    data[:, KeypointsNonFlatIndex.right_hand, 2] = z_pts_lh
    data[:, KeypointsNonFlatIndex.left_hand, 2] = z_pts_rh

    # Switch pose index
    # x_pts_rh, y_pts_rh, z_pts_rh = data[:, KeypointsNonFlatIndex.pose_right_hand, 0], \
    #     data[:, KeypointsNonFlatIndex.pose_right_hand, 1], \
    #     data[:, KeypointsNonFlatIndex.pose_right_hand, 2]
    
    # x_pts_lh, y_pts_lh, z_pts_lh = data[:, KeypointsNonFlatIndex.pose_left_hand, 0], \
    #     data[:, KeypointsNonFlatIndex.pose_left_hand, 1], \
    #     data[:, KeypointsNonFlatIndex.pose_left_hand, 2]

    # data[:, KeypointsNonFlatIndex.pose_right_hand, 0] = 1 - x_pts_lh
    # data[:, KeypointsNonFlatIndex.pose_left_hand, 0] = 1 - x_pts_rh
    
    # data[:, KeypointsNonFlatIndex.pose_right_hand, 1] = y_pts_lh
    # data[:, KeypointsNonFlatIndex.pose_left_hand, 1] = y_pts_rh

    # data[:, KeypointsNonFlatIndex.pose_right_hand, 2] = z_pts_lh
    # data[:, KeypointsNonFlatIndex.pose_left_hand, 2] = z_pts_rh


    return data


def apply_transform(transform, frame_keypoints):
    xy = [(np.max((pt[0], 0)), np.max((pt[1], 0))) for pt in frame_keypoints]

    augmented = transform(
        image=np.zeros((256, 256)),
        keypoints=xy,
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    
    augmented_keypoints = np.array([[kp[0], kp[1]] for kp in augmented['keypoints']])
    
    # Appending z axis data to the augmented dataset
    augmented_keypoints = np.stack(
        [augmented_keypoints[:, 0], augmented_keypoints[:, 1], frame_keypoints[:, 2]],
        axis=-1
    )
    
    return augmented, augmented_keypoints


def augment_keypoints(all_frames: np.ndarray, **kwargs) -> np.ndarray:
    """
    Augment keypoints using albumentations

    This function gets keypoints in the following format:
        (number_of_frames, TOTAL_PTS, 3)
        Which is 32X543x3 in our case.

    Args:
        all_frames (np.ndarray): Numpy array of the keypoints of all the frames
        Same shape as the incoming array

    Returns:
        Augmented numpy array of the keypoints
    """
    aug_frames = []

    transform = A.ReplayCompose([
            A.RandomScale(p=kwargs.get("p_scale"), scale_limit=0.5),
        ],
        keypoint_params=A.KeypointParams(format='xy')
    )

    augmented, augmented_keypoints = apply_transform(transform, all_frames[0])
    replay_params = augmented["replay"]
    aug_frames.append(augmented_keypoints)

    if not replay_params["applied"]:
        return all_frames

    # print(f"Augmentation params: {replay_params}")

    for frame_keypoints in all_frames[1:]:
        xy = [(np.max((pt[0], 0)), np.max((pt[1], 0))) for pt in frame_keypoints]

        augmented = A.ReplayCompose.replay(
            replay_params,
            image=np.zeros((16, 16)),
            keypoints=xy,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )

        augmented_keypoints = np.array([[kp[0], kp[1]] for kp in augmented['keypoints']])

        # Appending z axis data to the augmented dataset
        augmented_keypoints = np.stack(
            [augmented_keypoints[:, 0], augmented_keypoints[:, 1], frame_keypoints[:, 2]],
            axis=-1
        )

        aug_frames.append(augmented_keypoints)

    aug_frames = np.array(aug_frames)

    return aug_frames


def augment_np_keypoints(all_frames: np.ndarray, **kwargs) -> np.ndarray:
    """
    Augment keypoints in numnpy array

    Args:
        all_frames (np.ndarray): (n_frames, n_points, 3) -> Original coordinates

    Returns:
        np.ndarray: (n_frames, n_points, 3) -> Augmented coordinates
    """

    # Converting negative value points to 0 and multiplying by 512 for augmentations to work properly
    all_frames = np.maximum(all_frames.copy(), 0)
    all_frames = all_frames * 512

    # Define image widght and height
    image_width, image_height = kwargs.get("image_width", 512), kwargs.get("image_height", 512)
    # Center the keypoints
    center_point = np.array([image_width/2, image_height/2])
    augmented_keypoints = all_frames[:, :, :2] - center_point

    # Rotation degree
    roation_angle = np.random.uniform(-5, 5)
    angle = np.radians(roation_angle)
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])

    # Scaling factor
    scale_factor = np.random.uniform(0.5, 1.5)
    scaling_matrix = np.eye(2) * scale_factor

    logger.info(f"Rotation angle: {roation_angle}, scaling factor: {scale_factor}")

    if np.random.uniform() < kwargs.get("p_rotation", 0):
        augmented_keypoints = np.dot(augmented_keypoints, rotation_matrix.T)

    if np.random.uniform() < kwargs.get("p_scaling", 0):
        augmented_keypoints = np.dot(augmented_keypoints, scaling_matrix.T)

    # Add center point coordinates back to all keypoints
    augmented_keypoints = augmented_keypoints + center_point

    augmented_keypoints = np.stack([
        augmented_keypoints[:, :, 0],
        augmented_keypoints[:, :, 1],
        all_frames[:, :, 2]
    ], axis=-1)

    augmented_keypoints /= 512

    return augmented_keypoints


def add_jitter(all_frames:np.ndarray) -> np.ndarray:
    """
    Add jitter to keypoints
    """

    data = all_frames.copy()

    # Define the amount of jitter as a fraction of the image size
    jitter_fraction = 0.05
    jitter = np.random.uniform(-jitter_fraction, jitter_fraction, size=all_frames.shape)

    # Add the jitter to the original keypoints
    return data + jitter
