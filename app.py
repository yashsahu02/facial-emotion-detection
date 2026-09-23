import cv2
import numpy as np
import streamlit as st

from backend.pipeline import EmotionDetectionPipeline


st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="wide"
)


@st.cache_resource
def load_pipeline():
    return EmotionDetectionPipeline()


st.title("Facial Emotion Detection")
st.write(
    "Upload an image to detect faces and predict their emotions."
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error("Could not read the uploaded image.")

    else:

        pipeline = load_pipeline()

        results = pipeline.analyze(image)

        output_image = image.copy()

        for index, result in enumerate(results, start=1):

            box = result["box"]

            x = box["x"]
            y = box["y"]
            w = box["width"]
            h = box["height"]

            emotion = result["emotion"]
            confidence = result["confidence"]

            # Draw face bounding box
            cv2.rectangle(
                output_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Create label
            label = (
                f"Face {index}: "
                f"{emotion} "
                f"{confidence * 100:.1f}%"
            )

            # Calculate label size
            (label_width, label_height), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )
            )

            # Prevent label from going outside image
            label_y = max(y, label_height + baseline)

            # Draw label background
            cv2.rectangle(
                output_image,
                (x, label_y - label_height - baseline),
                (x + label_width, label_y),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                output_image,
                label,
                (x, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )


        # Display result
        if not results:

            st.warning("No faces detected in the image.")

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Uploaded Image",
                use_container_width=True
            )

        else:

            st.subheader("Detection Result")

            st.image(
                cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB),
                caption="Detected Faces and Emotions",
                use_container_width=True
            )


            st.subheader("Face Details")

            for index, result in enumerate(results, start=1):

                emotion = result["emotion"]
                confidence = result["confidence"]

                st.write(
                    f"**Face {index}:** "
                    f"{emotion.capitalize()} — "
                    f"{confidence * 100:.2f}% confidence"
                )