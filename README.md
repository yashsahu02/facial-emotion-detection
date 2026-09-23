# Facial Emotion Detection

A Python-based facial emotion detection application that detects faces in an uploaded image and predicts the facial expression of each detected face using a pretrained Vision Transformer (ViT) model.

The application uses OpenCV for face detection, Hugging Face Transformers for emotion classification, and Streamlit for the user interface.

## Overview

The system follows a two-stage approach:

1. Detect faces from the uploaded image using OpenCV's Haar Cascade classifier.
2. Crop each detected face and pass it to a pretrained facial expression recognition model.
3. Display the predicted emotion and confidence score for each detected face.
4. Draw bounding boxes and labels directly on the output image.

The application can process images containing multiple faces and handles images where no face is detected.

## Features

* Upload images through a Streamlit web interface
* Supports JPG, JPEG, PNG, and WEBP images
* Detects multiple faces in a single image
* Predicts the facial expression of each detected face
* Displays confidence scores for predictions
* Draws bounding boxes around detected faces
* Displays emotion labels directly on the detected faces
* Handles images with no detected faces
* Uses a pretrained emotion recognition model
* Modular backend architecture

## How It Works

The complete processing pipeline is:

```text
Uploaded Image
       │
       ▼
   Streamlit UI
       │
       ▼
   OpenCV Image
       │
       ▼
   Face Detection
   (Haar Cascade)
       │
       ▼
   Detected Faces
       │
       ▼
   Face Cropping
       │
       ▼
Pretrained ViT Emotion Model
       │
       ▼
Emotion + Confidence
       │
       ▼
Draw Bounding Boxes
       │
       ▼
Display Results
```

For an image containing multiple faces, each detected face is processed independently.

## Project Architecture

The project is organized into separate components for face detection, emotion prediction, application logic, testing, and the Streamlit interface.

```text
emotion_detection/
│
├── .streamlit/
│   └── config.toml
│
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── face_detector.py
│   ├── emotion_detector.py
│   └── pipeline.py
│
├── models/
│
├── test_images/
│   └── test.jpg
│
├── tests/
│   ├── test_face_detector.py
│   ├── test_emotion_detector.py
│   └── test_pipeline.py
│
├── app.py
├── pyproject.toml
└── README.md
```

### File Description

| File / Folder                    | Purpose                                                               |
| -------------------------------- | --------------------------------------------------------------------- |
| `app.py`                         | Streamlit frontend and application entry point                        |
| `backend/config.py`              | Stores the emotion model name and face detection configuration        |
| `backend/face_detector.py`       | Detects faces using OpenCV Haar Cascade                               |
| `backend/emotion_detector.py`    | Loads the pretrained emotion model and performs emotion prediction    |
| `backend/pipeline.py`            | Combines face detection and emotion prediction into a single workflow |
| `.streamlit/config.toml`         | Contains Streamlit server configuration                               |
| `tests/test_face_detector.py`    | Tests the face detection component                                    |
| `tests/test_emotion_detector.py` | Tests the emotion prediction component                                |
| `tests/test_pipeline.py`         | Tests the complete backend pipeline                                   |
| `test_images/`                   | Contains images used for testing                                      |
| `models/`                        | Reserved for locally stored model-related files if required           |
| `pyproject.toml`                 | Contains project metadata and dependencies                            |
| `README.md`                      | Project documentation                                                 |

## Installation

Follow the steps below to set up the project locally.

### 1. Clone the Repository

Clone the project from GitHub:

```bash
git clone https://github.com/yashsahu02/facial-emotion-detection
```

### 2. Move into the Project Directory

```bash
cd emotion_detection
```

### 3. Create the Virtual Environment

This project uses `uv` for Python environment and dependency management.

```bash
uv venv
```

### 4. Install Project Dependencies

Install all dependencies specified in `pyproject.toml`:

```bash
uv sync
```

This creates/uses the project's virtual environment and installs the required packages.

## Running the Application

After completing the installation steps, start the Streamlit application using:

```bash
uv run streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open the URL in your web browser.

## Using the Application

1. Open the Streamlit application in your browser.
2. Click **Choose an image**.
3. Upload an image in JPG, JPEG, PNG, or WEBP format.
4. The application detects faces in the uploaded image.
5. Each detected face is processed independently by the pretrained emotion recognition model.
6. The predicted emotion and confidence score are displayed on the image.
7. Bounding boxes are drawn around the detected faces.

If no face is detected, the application displays a warning and shows the uploaded image without emotion predictions.

## Running the Tests

The project contains separate test scripts for the main backend components.

### Test Face Detection

```bash
uv run python tests/test_face_detector.py
```

### Test Emotion Detection

```bash
uv run python tests/test_emotion_detector.py
```

### Test Complete Pipeline

```bash
uv run python tests/test_pipeline.py
```

These tests can be used to verify the individual components and the complete backend pipeline independently from the Streamlit interface.

## Technologies Used

* **Python** — Primary programming language
* **OpenCV** — Face detection and image processing
* **NumPy** — Image data handling and array operations
* **Pillow** — Image processing support
* **Hugging Face Transformers** — Pretrained model loading and inference
* **PyTorch** — Deep learning framework used by the pretrained model
* **Streamlit** — Web-based user interface
* **uv** — Python project and dependency management

## Pretrained Model

The project uses the following pretrained model:

`mo-thecreator/vit-Facial-Expression-Recognition`

The model is based on the Vision Transformer (ViT) architecture and is used for facial expression classification.

The project does not train a custom CNN or fine-tune the model. The pretrained model is loaded and used directly for inference.

## Face Detection

Face detection is performed using OpenCV's:

`haarcascade_frontalface_default.xml`

The Haar Cascade classifier identifies potential frontal faces in the uploaded image.

For each detected face, OpenCV returns a bounding box in the following format:

```text
(x, y, width, height)
```

The corresponding region is the
