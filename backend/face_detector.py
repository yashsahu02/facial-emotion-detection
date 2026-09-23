import cv2

from backend.config import (
    FACE_SCALE_FACTOR,
    FACE_MIN_NEIGHBORS,
    FACE_MIN_SIZE
)


class FaceDetector:

    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade.")

    def detect_faces(self, image):

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )

        return faces