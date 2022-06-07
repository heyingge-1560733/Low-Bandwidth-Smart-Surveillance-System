import struct
import ctypes
import pickle
import pickletools
import cv2
import mediapipe as mp
import sys
import time
import numpy as np
import os
from utils import *

_VISIBILITY_THRESHOLD = 0.5
TOTAL_LANDMARKS = 33

# Optimization flags
CONCATENATE_FLOATS = False
REDUCED_PRECISION = False
DROP_VISIBILITY = False
CHECK_VISIBILITY = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
# def get_next_frame_landmarks(mypath):
#     IMAGE_FILES = next(os.walk(mypath), (None, None, []))[2]  # [] if no file
#     BG_COLOR = (192, 192, 192) # gray
#     with mp_pose.Pose(
#         static_image_mode=False,
#         model_complexity=2,
#         enable_segmentation=True,
#         min_detection_confidence=0.5) as pose:
#       for idx, file in enumerate(IMAGE_FILES):
#         image = cv2.imread(mypath+file)
#         image_height, image_width, _ = image.shape
#         # Convert the BGR image to RGB before processing.
#         results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if not results.pose_landmarks:
#           continue
#         # print(
#         #     f'Nose coordinates: ('
#         #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
#         #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
#         # )
#         annotated_image = image.copy()
#         # Draw segmentation on the image.
#         # To improve segmentation around boundaries, consider applying a joint
#         # bilateral filter to "results.segmentation_mask" with "image".
#         condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#         bg_image = np.zeros(image.shape, dtype=np.uint8)
#         bg_image[:] = BG_COLOR
#         annotated_image = np.where(condition, annotated_image, bg_image)
#         # Draw pose landmarks on the image.
#         mp_drawing.draw_landmarks(
#             annotated_image,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#         # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#         # Plot pose world landmarks.
#         # mp_drawing.plot_landmarks(
#         #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
#         cv2.imshow("Burglary001", annotated_image)
#         cv2.waitKey(0)


def get_next_frame_landmarks():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            yield results.pose_world_landmarks

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            plt.close()
            # mp_drawing.plot_landmarks(
            #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    cap.release()


if __name__ == '__main__':
    optimization_level = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if optimization_level == 1:
        CONCATENATE_FLOATS = True
    elif optimization_level == 2:
        REDUCED_PRECISION = True
    elif optimization_level == 3:
        DROP_VISIBILITY = True
    elif optimization_level == 4:
        CHECK_VISIBILITY = True

    # initialize UDP socket
    sock = Sender()
    frame_id = 0

    # mypath = '/Volumes/Elements/Anomaly-Detection-Dataset/Anomaly-Videos-Part-2/Burglary/Burglary085/'
    for landmarks in get_next_frame_landmarks():
        if not CHECK_VISIBILITY:
            if not landmarks:
                continue

            # Baseline
            buf = pickle.dumps(landmarks)

            # Level 1 optimization
            if CONCATENATE_FLOATS:
                # float_list = []
                # buf = bytearray()
                data_size = struct.calcsize('<f')
                buf_size = struct.calcsize('<'+'f'*ATTRIBUTE_NUM*TOTAL_LANDMARKS)
                buf = ctypes.create_string_buffer(buf_size)

                for idx, landmark in enumerate(landmarks.landmark):
                    struct.pack_into('<ffff', buf, idx*data_size*ATTRIBUTE_NUM,
                                     landmark.x,
                                     landmark.y,
                                     landmark.z,
                                     landmark.visibility)
                # buf = (ctypes.c_double * len(float_list))()
                # buf[:] = float_list

            # Level 2 optimization
            if REDUCED_PRECISION:
                data_size = struct.calcsize('<e')
                buf_size = struct.calcsize('<'+'e'*ATTRIBUTE_NUM*TOTAL_LANDMARKS)
                buf = ctypes.create_string_buffer(buf_size)

                for idx, landmark in enumerate(landmarks.landmark):
                    struct.pack_into('<eeee', buf, idx*data_size*ATTRIBUTE_NUM,
                                     landmark.x,
                                     landmark.y,
                                     landmark.z,
                                     landmark.visibility)

            # Level 3 optimization
            if DROP_VISIBILITY:
                dim = 3
                data_size = struct.calcsize('<e')
                buf_size = struct.calcsize('<'+'e'*dim*TOTAL_LANDMARKS)
                buf = ctypes.create_string_buffer(buf_size)

                for idx, landmark in enumerate(landmarks.landmark):
                    struct.pack_into('<eee', buf, idx*data_size*dim,
                                     landmark.x,
                                     landmark.y,
                                     landmark.z)

        else:  # Level 4 optimization
            visible_num = 0

            if not landmarks:
                visible_landmarks = [False]*TOTAL_LANDMARKS
            else:
                visible_landmarks = [None]*TOTAL_LANDMARKS
                for idx, landmark in enumerate(landmarks.landmark):
                    if landmark.visibility < _VISIBILITY_THRESHOLD:
                        visible_landmarks[idx] = False
                    else:
                        visible_landmarks[idx] = True
                        visible_num += 1

            data_size = struct.calcsize('<e')
            buf_size = struct.calcsize('<'+'e'*3*visible_num)
            buf = ctypes.create_string_buffer(12+buf_size)  # 12 is frame ID + mode

            struct.pack_into('<H', buf, 0, frame_id % (2**16-1))

            # first 32
            for i in range(TOTAL_LANDMARKS // 4):
                mode = 0
                for j in range(4):
                    # not visible
                    if not visible_landmarks[i*4+j]:
                        mode |= (0b1 << (6-2*j))
                # print(bin(mode))
                # print(mode.to_bytes(1, 'little'))
                struct.pack_into('c', buf, 2+i, mode.to_bytes(1, 'little'))

            # 33th mode
            if not visible_landmarks[32]:
                struct.pack_into('c', buf, 10, b'\x40')
            else:
                struct.pack_into('c', buf, 10, b'\x00')

            if landmarks:
                i = 0
                for idx, landmark in enumerate(landmarks.landmark):
                    if visible_landmarks[idx]:
                        struct.pack_into('<eee', buf, 12 + i*data_size*3,
                                         landmark.x,
                                         landmark.y,
                                         landmark.z)
                        i += 1

        print(landmarks)
        # print(len(buf))

        sock.sendMessage(buf)
        # plot_landmarks_camera(landmarks, mp_pose.POSE_CONNECTIONS)
        # time.sleep(1)

        frame_id += 1
