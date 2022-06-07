import ctypes
import socket
import struct
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import *
import pickle

ATTRIBUTE_NUM = 4
TOTAL_LANDMARKS = 33
_VISIBILITY_THRESHOLD = 0.5

mp_pose = mp.solutions.pose

class Socket:
    def __init__(self, ip, port):
        self.udp_ip = ip
        self.udp_port = port
        self.start_message = b"Incoming info"
        self.end_message = b"End of message"

        self.sock = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP


class Sender(Socket):
    def __init__(self, ip="127.0.0.1", port=5005):
        Socket.__init__(self, ip, port)

    def sendMessage(self, message):
        self.sock.sendto(self.start_message, (self.udp_ip, self.udp_port))
        self.sock.sendto(message, (self.udp_ip, self.udp_port))
        self.sock.sendto(self.end_message, (self.udp_ip, self.udp_port))


class Listener(Socket):
    def __init__(self, ip="127.0.0.1", port=5005):
        Socket.__init__(self, ip, port)
        self.sock.bind((ip, port))
        self.frame_count = 0
        self.total_size = 0

    def listen(self):
        message = bytearray()
        while True:
            data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
            if self.frame_count == 30:
                print("throughput: %s kbps" % (self.total_size * 8 / 1000))
                self.frame_count = 0
                self.total_size = 0

            self.total_size += len(data)
            if data == self.start_message:
                message = bytearray()
            elif data == self.end_message:
                # print("received frame of %s bytes" % len(message)) # uncomment for frame sizes
                yield message
                self.frame_count += 1
            else:
                message.extend(data)


def display_message(data, dim=4):
    fmt = []

    for i in range(len(data) // dim):
        if dim == 4:
            fmt.append(
                'landmark {\n' +
                '  x: ' + str(data[dim*i]) + '\n'
                '  y: ' + str(data[dim*i+1]) + '\n'
                '  z: ' + str(data[dim*i+2]) + '\n'
                '  visibility: ' + str(data[dim*i+3]) + '\n'
                '}')
        elif dim == 3:
            fmt.append(
                'landmark {\n' +
                '  x: ' + str(data[dim*i]) + '\n'
                '  y: ' + str(data[dim*i+1]) + '\n'
                '  z: ' + str(data[dim*i+2]) + '\n'
                '}')

    print('\n'.join(fmt))


def deserialize_message(message):
    return pickle.loads(message)


def unpack_message(message, fmt, endianness='little'):
    if endianness == 'little':
        endian = '<'
    elif endianness == 'big':
        endian = '>'
    else:
        raise ValueError('Invalid endianness')

    data_size = struct.calcsize(endian+fmt)
    data_num = len(message) // data_size  # number of values
    return struct.unpack_from(endian+fmt*data_num, message, 0)


def decode_message(message, fmt, endianness='little'):
    if endianness == 'little':
        endian = '<'
    elif endianness == 'big':
        endian = '>'
    else:
        raise ValueError('Invalid endianness')

    frame_id = struct.unpack_from('<H', message, 0)[0]

    visible_landmarks = [None]*TOTAL_LANDMARKS
    # first 32
    # print(message)
    for i in range(TOTAL_LANDMARKS // 4):
        mode = struct.unpack_from('c', message, 2+i)[0]
        # print(bin(ord(mode)))
        for j in range(4):
            # not visible
            mode_masked = ord(mode) & (0b11 << (6-2*j))
            # print(mode_masked)
            visible_landmarks[i*4+j] = (mode_masked == 0)
    # 33th mode
    visible_landmarks[32] = (struct.unpack_from('c', message, 10) == 0)

    data = []
    data_size = struct.calcsize(endian+fmt)
    data_num = (len(message)-12) // data_size  # number of values
    data_short = struct.unpack_from(endian+fmt*data_num, message, 12)

    j = 0
    for i in range(TOTAL_LANDMARKS):
        if visible_landmarks[i]:
            data.extend(data_short[3*j:3*j+3])
            data.append(1)
            j += 1
        else:
            data.extend([0,0,0,0])

    return frame_id, tuple(data)


def _normalize_color(color):
    return tuple(v / 255. for v in color)


# modified plot_landmarks API, works for optimization level 1 and 2
def plot_landmarks_server(landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: DrawingSpec = DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: DrawingSpec = DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10):
    """Plot the landmarks and the connections in matplotlib 3d.

    Args:
      landmark_list: A normalized landmark list proto message to be plotted.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
      landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
      connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
      elevation: The elevation from which to view the plot.
      azimuth: the azimuth angle to rotate the plot.
    Raises:
      ValueError: If any connetions contain invalid landmark index.
    """
    plt.close()
    if not landmark_list:
        return
    plt.figure(figsize=(10, 10))
    plt.ion()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx in range(0, len(landmark_list), ATTRIBUTE_NUM):
        x = landmark_list[idx]
        y = landmark_list[idx+1]
        z = landmark_list[idx+2]
        visibility = landmark_list[idx+3]
        if visibility < _VISIBILITY_THRESHOLD:
            continue
        ax.scatter3D(
            xs=[-z],
            ys=[x],
            zs=[-y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness)
        plotted_landmarks[idx] = (-z, x, -y)
    if connections:
        num_landmarks = len(landmark_list) // ATTRIBUTE_NUM
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    plt.show()

def plot_landmarks_camera(landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: DrawingSpec = DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: DrawingSpec = DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10):
    """Plot the landmarks and the connections in matplotlib 3d.

    Args:
      landmark_list: A normalized landmark list proto message to be plotted.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
      landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
      connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
      elevation: The elevation from which to view the plot.
      azimuth: the azimuth angle to rotate the plot.
    Raises:
      ValueError: If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    plt.figure(figsize=(10, 10))
    plt.ion()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness)
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    plt.show()
