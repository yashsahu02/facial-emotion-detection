from transformers import pipeline

from backend.config import EMOTION_MODEL_NAME


class EmotionDetector:

    def __init__(self):

        print("Loading emotion model...")

        self.classifier = pipeline(
            "image-classification",
            model=EMOTION_MODEL_NAME
        )

        print("Emotion model loaded successfully!")

    def predict(self, image):

        results = self.classifier(image)

        best_result = results[0]

        return {
            "emotion": best_result["label"],
            "confidence": best_result["score"]
        }