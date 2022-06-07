import threading
import time
from utils import *
from classifier import *
import sys
from collections import deque
import mediapipe as mp
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

BASELINE = False
CONCATENATE_FLOATS = False
REDUCED_PRECISION = False
CHECK_VISIBILITY = False

POSES_QUEUE = deque()
CONSECUTIVE = 5
TARGET = 'burglary'

def wait_for_event(e):
    target_counter = 0
    while True:
        if len(POSES_QUEUE) == 0:
            # logging.debug('wait_for_event starting')
            event_is_set = e.wait()
            # logging.debug('event set: %s', event_is_set)

        class_name = classifier.classify(POSES_QUEUE.popleft())
        target_counter = target_counter + 1 if class_name == TARGET else 0
        logging.debug('target counter: %d', target_counter)
        logging.debug('waiting poses: %d', len(POSES_QUEUE))


if __name__ == '__main__':
    optimization_level = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # initializations
    if optimization_level == 0:
        BASELINE = True
    if optimization_level == 1:
        CONCATENATE_FLOATS = True
    if optimization_level == 2:
        REDUCED_PRECISION = True
    if optimization_level == 3:
        DROP_VISIBILITY = True
    if optimization_level == 4:
        CHECK_VISIBILITY = True

    csvs_dir = '/Users/freddie/Homeworks/Capstone/code/colab/combined/poses_csvs_out (before further filter)'
    pose_embedder = FullBodyPoseEmbedder()
    classifier = PoseClassifier(pose_samples_folder=csvs_dir,
                                pose_embedder=pose_embedder)
    e = threading.Event()
    t1 = threading.Thread(target=wait_for_event, name='classifier', args=(e,))
    t1.start()

    # start listening
    listener = Listener()
    for message in listener.listen():
        if BASELINE:
            message = deserialize_message(message)
            # print(message)
        elif CONCATENATE_FLOATS:
            message = unpack_message(message, 'f', endianness='little')
            # display_message(message)
            message_np = np.asarray(message).reshape((-1,ATTRIBUTE_NUM))
        elif REDUCED_PRECISION:
            message = unpack_message(message, 'e', endianness='little')
            # display_message(message)
            message_np = np.asarray(message).reshape((-1,ATTRIBUTE_NUM))
        elif DROP_VISIBILITY:
            message = unpack_message(message, 'e', endianness='little')
            # display_message(message, 3)
            message_np = np.asarray(message).reshape((-1,3))
        elif CHECK_VISIBILITY:
            frame_id, message = decode_message(message, 'e', endianness='little')
            # display_message(message)
            message_np = np.asarray(message).reshape((-1,ATTRIBUTE_NUM))

        # mp_drawing.plot_landmarks(
        #     pickle.loads(message), mp_pose.POSE_CONNECTIONS)

        POSES_QUEUE.append(message_np)
        # logging.debug('Waiting before calling Event.set()')
        e.set()
        # logging.debug('Event is set')
        e.clear()
