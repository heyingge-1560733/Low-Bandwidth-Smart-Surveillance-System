import csv
import numpy as np
import os
import random


class PoseClassifier(object):
    """Classifies pose landmarks."""

    def __init__(self,
                 pose_samples_folder,
                 pose_embedder,
                 file_extension='csv',
                 file_separator=',',
                 n_landmarks=33,
                 n_dimensions=3,
                 axes_weights=(1., 1., 0.2)):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._axes_weights = axes_weights

        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                     file_extension, file_separator,
                                                     n_landmarks, n_dimensions,
                                                     pose_embedder)

        print('Successfully loaded {} poses.'.format(len(self._pose_samples)))

    def _load_pose_samples(self, pose_samples_folder,
                           file_extension, file_separator,
                           n_landmarks, n_dimensions,
                           pose_embedder):
        """Loads pose samples from a given folder.

        Required folder structure:
          train/
            normal.csv
            burglary.csv
            ...
          test/
            normal.csv
            burglary.csv
            ...

        Required CSV structure:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
          ...
        """
        # Each file in the folder represents one pose class.
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

            # Parse CSV.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
                    pose_samples.append(PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),
                    ))

        return pose_samples

    def classify(self,
                 pose,
                 top_n_by_max_distance=25,
                 top_n_by_mean_distance=5):
        # Find nearest poses for the target one.
        pose_landmarks = pose.copy()
        pose_classification = self.__call__(pose_landmarks, top_n_by_max_distance, top_n_by_mean_distance)
        class_names = [class_name for class_name, count in pose_classification.items() if count == max(pose_classification.values())]

        # Sample is an outlier if nearest poses have different class or more than
        # one pose class is detected as nearest.
        return random.choice(class_names)

    #   def __call__(self, pose_landmarks, top_n_by_max_distance, top_n_by_mean_distance, axes_weights):
    #     """Classifies given pose.

    #     Classification is done in two stages:
    #       * First we pick top-N samples by MAX distance. It allows to remove samples
    #         that are almost the same as given pose, but has few joints bent in the
    #         other direction.
    #       * Then we pick top-N samples by MEAN distance. After outliers are removed
    #         on a previous step, we can pick samples that are closes on average.

    #     Args:
    #       pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

    #     Returns:
    #       Dictionary with count of nearest pose samples from the database. Sample:
    #         {
    #           'normal': 8,
    #           'burglary': 2,
    #         }
    #     """
    #     # Check that provided and target poses have the same shape.
    #     assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

    #     # Get given pose embedding.
    #     pose_embedding = self._pose_embedder(pose_landmarks)
    #     flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

    #     # Filter by max distance.
    #     #
    #     # That helps to remove outliers - poses that are almost the same as the
    #     # given one, but has one joint bent into another direction and actually
    #     # represnt a different pose class.
    #     max_dist_heap = []
    #     for sample_idx, sample in enumerate(self._pose_samples):
    #       max_dist = min(
    #           np.max(np.abs(sample.embedding - pose_embedding) * axes_weights),
    #           np.max(np.abs(sample.embedding - flipped_pose_embedding) * axes_weights),
    #       )
    #       max_dist_heap.append([max_dist, sample_idx])

    #     max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
    #     max_dist_heap = max_dist_heap[:top_n_by_max_distance]

    #     # Filter by mean distance.
    #     #
    #     # After removing outliers we can find the nearest pose by mean distance.
    #     mean_dist_heap = []
    #     for _, sample_idx in max_dist_heap:
    #       sample = self._pose_samples[sample_idx]
    #       mean_dist = min(
    #           np.mean(np.abs(sample.embedding - pose_embedding) * axes_weights),
    #           np.mean(np.abs(sample.embedding - flipped_pose_embedding) * axes_weights),
    #       )
    #       mean_dist_heap.append([mean_dist, sample_idx])

    #     mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
    #     mean_dist_heap = mean_dist_heap[:top_n_by_mean_distance]

    #     # Collect results into map: (class_name -> n_samples)
    #     class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
    #     result = {class_name: class_names.count(class_name) for class_name in set(class_names)}

    #     return result

    def __call__(self, pose_landmarks, top_n_by_max_distance, top_n_by_mean_distance):
        """Classifies given pose.

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.

        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns:
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'normal': 8,
              'burglary': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represnt a different pose class.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding)),
                np.max(np.abs(sample.embedding - flipped_pose_embedding)),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:top_n_by_max_distance]

        # Filter by mean distance.
        #
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding)),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding)),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:top_n_by_mean_distance]

        # Collect results into map: (class_name -> n_samples)
        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = {class_name: class_names.count(class_name) for class_name in set(class_names)}

        return result

class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name
        self.embedding = embedding

class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 31,
            'left_foot_index': 32, 'right_foot_index': 33
        }

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names['left_hip']]
        right_hip = landmarks[self._landmark_names['right_hip']]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names['left_hip']]
        right_hip = landmarks[self._landmark_names['right_hip']]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names['left_shoulder']]
        right_shoulder = landmarks[self._landmark_names['right_shoulder']]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        # embedding = np.array([
        #     # One joint.

        #     self._get_distance(
        #         self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
        #         self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

        #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
        #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

        #     self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

        #     self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
        #     self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

        #     # Two joints.

        #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

        #     # Four joints.

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        #     # Five joints.

        #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
        #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        #     # Cross body.

        #     self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
        #     self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

        #     self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
        #     self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

        #     # Body bent direction.

        #     self._get_distance(
        #         self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
        #         landmarks[self._landmark_names['left_hip']]),
        #     self._get_distance(
        #         self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
        #         landmarks[self._landmark_names['right_hip']]),
        # ])

        embedding = np.array([
            # One side.
            self._get_angle_by_names(landmarks, 'left_wrist', 'left_elbow', 'left_shoulder'),
            self._get_angle_by_names(landmarks, 'right_wrist', 'right_elbow', 'right_shoulder'),
            self._get_angle_by_names(landmarks, 'left_shoulder', 'left_hip', 'left_knee'),
            self._get_angle_by_names(landmarks, 'right_shoulder', 'right_hip', 'right_knee'),
            self._get_angle_by_names(landmarks, 'left_hip', 'left_knee', 'left_ankle'),
            self._get_angle_by_names(landmarks, 'right_hip', 'right_knee', 'right_ankle'),
            self._get_angle_by_names(landmarks, 'left_elbow', 'left_shoulder', 'left_hip'),
            self._get_angle_by_names(landmarks, 'right_elbow', 'right_shoulder', 'right_hip'),

            # Cross body.
            self._get_angle_by_names(landmarks, 'right_shoulder', 'left_shoulder', 'left_hip'),
            self._get_angle_by_names(landmarks, 'left_shoulder', 'right_shoulder', 'right_hip'),
            self._get_angle_by_names(landmarks, 'left_shoulder', 'left_hip', 'right_hip'),
            self._get_angle_by_names(landmarks, 'right_shoulder', 'right_hip', 'left_hip'),
            self._get_angle_by_names(landmarks, 'left_hip', 'right_hip', 'right_knee'),
            self._get_angle_by_names(landmarks, 'right_hip', 'left_hip', 'left_knee'),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names[name_from]]
        lmk_to = landmarks[self._landmark_names[name_to]]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names[name_from]]
        lmk_to = landmarks[self._landmark_names[name_to]]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from

    def _get_angle_by_names(self, landmarks, name_from, name_center, name_to):
        lmk_from = landmarks[self._landmark_names[name_from]]
        lmk_center = landmarks[self._landmark_names[name_center]]
        lmk_to = landmarks[self._landmark_names[name_to]]
        return self._get_angle(lmk_from, lmk_center, lmk_to)

    def _get_angle(self, lmk_from, lmk_center, lmk_to):
        vec_from = (lmk_from - lmk_center) * (1., 1., 0.2)
        vec_to = (lmk_to - lmk_center) * (1., 1., 0.2)

        return np.dot(vec_from, vec_to) / (np.linalg.norm(vec_from) * np.linalg.norm(vec_to))