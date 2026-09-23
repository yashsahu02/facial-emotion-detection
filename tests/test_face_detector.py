import cv2

from backend.face_detector import FaceDetector


IMAGE_PATH = "test_images/img1.webp"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

face_detector = FaceDetector()

faces = face_detector.detect_faces(image)

print(f"Number of faces detected: {len(faces)}")

for index, (x, y, w, h) in enumerate(faces, start=1):
    print(
        f"Face {index}: "
        f"x={x}, y={y}, width={w}, height={h}"
    )