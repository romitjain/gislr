"""Definition of all keypoints go here"""
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

TOTAL_PTS = 543
mp_holistic = mp.solutions.holistic

@dataclass
class KeypointsIndex():
    """
    Index points in flattenned numpy array
    """
    face=(0, 468*3)
    face_x_index = [x for x in range(face[0], face[1], 3)]
    face_y_index = [x+1 for x in range(face[0], face[1], 3)]
    face_z_index = [x+2 for x in range(face[0], face[1], 3)]
    left_hand = (468*3, 489*3)
    left_hand_x_index = [x for x in range(left_hand[0], left_hand[1], 3)]
    left_hand_y_index = [x+1 for x in range(left_hand[0], left_hand[1], 3)]
    left_hand_z_index = [x+2 for x in range(left_hand[0], left_hand[1], 3)]
    pose = (489*3, 522*3)
    pose_x_index = [x for x in range(pose[0], pose[1], 3)]
    pose_y_index = [x+1 for x in range(pose[0], pose[1], 3)]
    pose_z_index = [x+2 for x in range(pose[0], pose[1], 3)]
    right_hand = (522*3, 543*3)
    right_hand_x_index = [x for x in range(right_hand[0], right_hand[1], 3)]
    right_hand_y_index = [x+1 for x in range(right_hand[0], right_hand[1], 3)]
    right_hand_z_index = [x+2 for x in range(right_hand[0], right_hand[1], 3)]
    all_x_index = [x for x in range(0, TOTAL_PTS, 3)]
    all_y_index = [x+1 for x in range(0, TOTAL_PTS, 3)]
    all_z_index = [x+2 for x in range(0, TOTAL_PTS, 3)]


@dataclass
class KeypointsNonFlatIndex():
    """
    Index points in non-flattenned numpy array
    """
    face=np.arange(0, 468)
    left_hand = np.arange(468, 489)
    pose = np.arange(489, 522)
    right_hand = np.arange(522, 543)

    pose_left_hand = [
        mp_holistic.PoseLandmark.LEFT_WRIST.value,
        mp_holistic.PoseLandmark.LEFT_PINKY.value,
        mp_holistic.PoseLandmark.LEFT_INDEX.value,
        mp_holistic.PoseLandmark.LEFT_THUMB.value
    ]

    pose_right_hand = [x+1 for x in pose_left_hand]

    lips = set()

    for elem in mp.solutions.face_mesh_connections.FACEMESH_LIPS:
        lips.add(elem[0])
        lips.add(elem[1])

    eyes = set()

    for elem in mp.solutions.face_mesh_connections.FACEMESH_RIGHT_EYE:
        eyes.add(elem[0])
        eyes.add(elem[1])

    for elem in mp.solutions.face_mesh_connections.FACEMESH_LEFT_EYE:
        eyes.add(elem[0])
        eyes.add(elem[1])

    for elem in mp.solutions.face_mesh_connections.FACEMESH_IRISES:
        eyes.add(elem[0])
        eyes.add(elem[1])

    # Class labels for all 543 points
    class_labels = []

    class_labels += ["face"]*len(face)
    class_labels += ["left_hand"]*len(left_hand)
    class_labels += ["pose"]*len(pose)
    class_labels += ["right_hand"]*len(right_hand)


connections_type = dict(
    face=mp_holistic.FACEMESH_TESSELATION,
    left_hand=mp_holistic.HAND_CONNECTIONS,
    right_hand=mp_holistic.HAND_CONNECTIONS,
    pose=mp_holistic.POSE_CONNECTIONS
)

index_labels = dict(
    face=(0, 468),
    left_hand=(468, 489),
    pose=(489, 522),
    right_hand=(522, 543)
)
