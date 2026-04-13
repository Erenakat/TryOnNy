import sys
import cv2
import mediapipe as mp
import numpy as np

def main(image_path: str):
    print("Loading image:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Failed to load image")
        return

    print("Shape:", image.shape)
    print("Dtype:", image.dtype)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("❌ No landmarks detected")
    else:
        print("✅ Landmarks detected:", len(results.pose_landmarks.landmark))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.test_pose path/to/image.jpg")
    else:
        main(sys.argv[1])