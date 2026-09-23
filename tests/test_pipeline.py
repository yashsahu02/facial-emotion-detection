import cv2

from backend.pipeline import EmotionDetectionPipeline


IMAGE_PATH = "test_images/earth_image.webp"


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


pipeline = EmotionDetectionPipeline()

results = pipeline.analyze(image)


print("\nEmotion Detection Results:")


if not results:

    print("No faces detected.")

else:

    for index, result in enumerate(results, start=1):

        box = result["box"]

        print(f"\nFace {index}")
        print(
            f"Bounding Box: "
            f"x={box['x']}, "
            f"y={box['y']}, "
            f"width={box['width']}, "
            f"height={box['height']}"
        )

        print(
            f"Emotion: {result['emotion']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )