# Project Requirements & Technical (PRT) Document

# Human Anomaly Detection Using Computer Vision

**Version:** 1.0  
**Project Type:** Computer Vision / Machine Learning  
**Architecture:** Standalone module designed for later integration with an ANN-based object detection project.

---

## 1. Project Overview

The Human Anomaly Detection module is a computer-vision and machine-learning system intended to identify unusual or potentially unsafe human activity from images or video.

The first implementation is deliberately kept as a separate project folder so that its detection and classification pipeline can later be connected to an ANN/object-detection system.

---

## 2. Problem Statement

Manual monitoring of surveillance video is time-consuming and can miss events because of human fatigue or limited attention.

The proposed system processes visual input and classifies detected human regions as **normal or anomalous** according to the classes represented in the training dataset.

---

## 3. Objectives

- Detect human activity from image or video using computer vision.
- Classify the relevant human region into normal or anomaly categories.
- Provide a confidence score and clear prediction for each analyzed frame/person.
- Keep preprocessing, inference and model code modular for later reuse.
- Allow an external object detector to pass a cropped person image directly to the anomaly detector.
- Evaluate the model using accuracy, precision, recall, F1-score and a confusion matrix.

---

## 4. Scope

The standalone module covers:

- Dataset preparation
- Preprocessing
- Model training
- Validation
- Testing
- Image/video inference
- Result visualization

The system does **not** attempt to determine intent, identity, or legal culpability.

The definition of an anomaly is dataset-dependent and must be explicitly documented for the selected dataset.

---

## 5. Proposed System

```text
Image / Camera / Video
        |
        v
     OpenCV
        |
        v
Human / Person Region
        |
        v
   Preprocessing
        |
        v
    ML Model
        |
        v
Normal / Anomaly
        |
        v
 Display / Alert
```

---

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Load images or frames from a configured dataset or video source. |
| FR-02 | Preprocess input to the model's required size and numerical range. |
| FR-03 | Train a binary or multi-class anomaly classifier from labeled data. |
| FR-04 | Run inference on an individual image, video frame, or cropped person region. |
| FR-05 | Return predicted class and confidence score. |
| FR-06 | Save trained model and evaluation artifacts. |
| FR-07 | Generate evaluation metrics and confusion matrix. |
| FR-08 | Expose a reusable prediction function for future ANN/object-detector integration. |

---

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | Inference should be lightweight enough for near-real-time experimentation on supported hardware. |
| Reliability | Invalid images and unavailable model files should produce controlled errors. |
| Maintainability | Training, preprocessing and prediction logic should remain in separate modules. |
| Scalability | The classifier should support additional anomaly classes if the dataset is expanded. |
| Usability | Prediction output should be understandable to a student/developer using the module. |
| Integration | The prediction interface should accept an image/person crop without depending on the upstream detector. |

---

## 8. Hardware & Software Requirements

| Item | Specification |
|---|---|
| OS | Windows / Linux / macOS |
| Language | Python 3.10+ recommended |
| Libraries | OpenCV, NumPy, TensorFlow/Keras or PyTorch, scikit-learn, Matplotlib |
| Development | VS Code, Jupyter Notebook or Google Colab |
| GPU | Optional; useful for CNN training on larger image datasets |
| RAM | 8 GB minimum recommended |
| Storage | Depends on the selected dataset and trained models |

---

## 9. Dataset Requirements

The dataset should contain labeled examples representing the chosen definition of normal and anomalous human activity.

Train, validation and test sets should be separated to reduce data leakage.

If video frames are extracted, frames from the same original video should preferably remain within the same split.

### Suggested Dataset Structure

```text
dataset/
├── train/
│   ├── normal/
│   └── anomaly/
├── validation/
│   ├── normal/
│   └── anomaly/
└── test/
    ├── normal/
    └── anomaly/
```

---

## 10. ML Methodology

1. Collect/select a suitable human-activity anomaly dataset.
2. Clean data and remove corrupt or irrelevant samples.
3. Resize images and normalize pixel values.
4. Apply appropriate augmentation to training data only.
5. Train a CNN-based classifier or compatible ANN-style image classifier.
6. Use validation data to monitor generalization and reduce overfitting.
7. Evaluate on an unseen test set.
8. Save the trained model and preprocessing configuration.

---

## 11. Model Input / Output

| Component | Specification |
|---|---|
| Input | RGB/BGR image or cropped person region |
| Image Processing | Resize and normalize according to the trained model |
| Output Class | Normal or Anomaly |
| Confidence | Numeric probability/confidence associated with prediction |
| API Result | Dictionary/object containing label, confidence and optional metadata |

---

## 12. Proposed Folder Structure

```text
human_anomaly_detection/
│
├── dataset/
│   ├── train/
│   │   ├── normal/
│   │   └── anomaly/
│   ├── validation/
│   │   ├── normal/
│   │   └── anomaly/
│   └── test/
│       ├── normal/
│       └── anomaly/
│
├── models/
│   └── anomaly_model.h5
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── anomaly_detector.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── predictions/
│
├── app.py
├── requirements.txt
├── README.md
├── PRT.md
└── SSD.md
```

---

## 13. Integration Plan With ANN Object Detection

During integration, the existing ANN/object detector will identify a person and produce a bounding box.

The bounding box will be cropped from the frame and passed to the anomaly detector.

The anomaly detector will return the classification and confidence.

The combined application can then overlay both the person bounding box and anomaly status.

### Target Interface

```python
result = anomaly_detector.predict(person_image)
```

### Expected Result

```python
{
    "label": "Anomaly",
    "confidence": 0.91
}
```

The exact result format can be expanded later to include:

- Timestamp
- Person ID
- Bounding box
- Frame number

---

## 14. Testing Requirements

- Test normal and anomaly images independently.
- Test different image sizes and aspect ratios.
- Test video frames with multiple people where supported.
- Verify that model loading and prediction errors are handled.
- Measure accuracy, precision, recall and F1-score.
- Inspect false positives and false negatives.
- After integration, test that the object detector's crop is correctly passed to the anomaly model.

---

## 15. Expected Outcome

A working standalone CV/ML module capable of classifying human regions according to the selected anomaly definition, producing measurable evaluation results, and exposing a reusable prediction interface for the later ANN object-detection project.

---

## 16. Future Scope

- Multi-class anomaly recognition.
- Temporal/video-based anomaly detection using CNN-LSTM or other sequence models.
- Real-time camera streaming.
- Configurable alert thresholds.
- Deployment as a Flask/FastAPI service.
- Integration with the existing ANN object detector and dashboard.
