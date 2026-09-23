# from transformers import pipeline

# # MODEL_NAME = "HardlyHumans/Facial-expression-detection"
# MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

# print("Loading emotion model...")

# emotion_classifier = pipeline(
#     "image-classification",
#     model=MODEL_NAME
# )

# print("Model loaded successfully!")


# -------------------------- Version 2 --------------------------
from transformers import pipeline

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"
IMAGE_PATH = "test_images/img1.webp"

print("Loading emotion model...")

emotion_classifier = pipeline(
    "image-classification",
    model=MODEL_NAME
)

print("Model loaded successfully!")
print("\nRunning prediction...")

results = emotion_classifier(IMAGE_PATH)

print("\nPredictions:")

for result in results:
    print(
        f"{result['label']}: "
        f"{result['score'] * 100:.2f}%"
    )