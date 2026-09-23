import cv2

from backend.emotion_detector import EmotionDetector


IMAGE_PATH = "test_images/img1.webp"


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


emotion_detector = EmotionDetector()

result = emotion_detector.predict(image)


print("\nPrediction:")
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")