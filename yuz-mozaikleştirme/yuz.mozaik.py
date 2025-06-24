from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        koordinat = []
        for landmark in face_landmarks:
            koordinat.append(str(round(landmark.x, 4)))
            koordinat.append(str(round(landmark.y, 4)))

        koordinat = ",".join(koordinat)
        koordinat += f",{etiket}\n"
        with open("veriseti.csv", "a") as f:
            f.write(koordinat)

       
        h, w, _ = annotated_image.shape
        x_list = [int(landmark.x * w) for landmark in face_landmarks]
        y_list = [int(landmark.y * h) for landmark in face_landmarks]

        x_min, x_max = max(min(x_list) - 10, 0), min(max(x_list) + 10, w)
        y_min, y_max = max(min(y_list) - 10, 0), min(max(y_list) + 10, h)

        face_region = annotated_image[y_min:y_max, x_min:x_max]
        if face_region.size > 0:
            small = cv2.resize(face_region, (16, 16), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
            annotated_image[y_min:y_max, x_min:x_max] = mosaic
        

    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))
    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

etiket = "happy"
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

kamera = cv2.VideoCapture(0)
while kamera.isOpened():
    basari, frame = kamera.read()
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("yuz", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord('e') or key == ord('E'):
            exit(0)
