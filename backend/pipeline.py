from backend.face_detector import FaceDetector
from backend.emotion_detector import EmotionDetector


class EmotionDetectionPipeline:

    def __init__(self):
        self.face_detector = FaceDetector()
        self.emotion_detector = EmotionDetector()

    def analyze(self, image):

        if image is None:
            raise ValueError("Invalid image.")

        faces = self.face_detector.detect_faces(image)

        results = []

        for x, y, w, h in faces:

            face_crop = image[y:y + h, x:x + w]

            emotion_result = self.emotion_detector.predict(
                face_crop
            )

            results.append({
                "box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "emotion": emotion_result["emotion"],
                "confidence": float(emotion_result["confidence"])
            })

        return results