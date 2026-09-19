# Software / System Specification Document (SSD)

# Human Anomaly Detection Using Computer Vision

**Version:** 1.0  
**Project Type:** Computer Vision / Machine Learning  
**Architecture:** Standalone modular anomaly-detection component.

---

## 1. System Purpose

This Software/System Specification Document defines the software architecture, modules, data flow, interfaces, processing rules, testing requirements and integration contract for a Human Anomaly Detection component.

The system is designed as a separate module so that it can later be merged with an ANN-based object detection project.

---

## 2. System Architecture

```text
Video / Image Source
        |
        v
      OpenCV
        |
        v
Person Detection / Crop
        |
        v
   Preprocessing
        |
        v
  Anomaly ML Model
        |
        v
   Post-processing
        |
        v
 Result / Alert
```

---

## 3. Module Specifications

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Image loading, resizing, color conversion, normalization and optional augmentation utilities. |
| `train.py` | Dataset loading, model creation, training, validation, checkpointing and metric generation. |
| `predict.py` | Command-line or application-level inference on images/video frames. |
| `anomaly_detector.py` | Reusable detector class/function that loads the trained model and exposes `predict()`. |
| `app.py` | Optional user-facing entry point for camera/video/image inference. |
| `models/` | Stores trained model files and associated configuration. |
| `results/` | Stores plots, confusion matrix, metrics and prediction outputs. |

---

## 4. Data Flow

1. Acquire an image or video frame.
2. Identify or receive a person region.
3. Crop and resize the person region.
4. Normalize the image.
5. Run model inference.
6. Convert model output to a label and confidence.
7. Apply the configured decision threshold.
8. Return or display the result.
9. Optionally log the prediction.

---

## 5. Input Specification

| Input | Specification |
|---|---|
| Image | JPEG/PNG or OpenCV-compatible image. |
| Video | Common OpenCV-readable video formats or camera stream. |
| Person Crop | Image region supplied by an upstream object detector. |
| Pixel Format | Must remain consistent between training and inference. |
| Model Input | Fixed dimensions determined by the selected model architecture. |

---

## 6. Output Specification

| Output | Description |
|---|---|
| `label` | Predicted category such as Normal or Anomaly. |
| `confidence` | Numeric confidence/probability associated with prediction. |
| `visual_output` | Optional bounding box/label overlay when integrated with object detection. |
| `alert` | Optional event generated when confidence exceeds a configured threshold. |

---

## 7. Model Specification

The initial implementation may use a CNN image classifier because anomaly classification is based on visual features.

The exact architecture should be selected according to:

- Dataset size
- Computational resources
- Required inference speed
- Number of anomaly classes
- Desired accuracy

Transfer learning can be considered when the available dataset is limited.

---

## 8. Training Specification

- Use separate train, validation and test splits.
- Normalize inputs consistently.
- Apply augmentation only to training samples.
- Track training and validation loss/metrics.
- Use early stopping or checkpointing where appropriate.
- Evaluate on the held-out test set after model selection.
- Save the final model.
- Record preprocessing assumptions.

---

## 9. Inference Specification

The anomaly detector must provide a small and stable interface so it can be called by another application.

The upstream ANN object detector should not need to know the internal anomaly-model architecture.

### Example Interface

```python
result = anomaly_detector.predict(person_image)
```

### Example Response

```python
{
    "label": "Anomaly",
    "confidence": 0.91
}
```

---

## 10. Configuration

| Parameter | Description |
|---|---|
| `MODEL_PATH` | Path to the trained anomaly model. |
| `IMAGE_SIZE` | Width/height expected by the model. |
| `THRESHOLD` | Decision threshold for anomaly classification. |
| `CLASS_NAMES` | Mapping between model outputs and human-readable labels. |
| `SOURCE` | Image, video file or camera index. |

---

## 11. Error Handling

The system should handle errors without crashing the complete application.

- If the model file is missing, report a clear model-loading error.
- If an input image cannot be decoded, report the input error.
- If the image is empty or invalid, do not run inference.
- If preprocessing fails, return a controlled error.
- During integration, handle missing person crops without crashing the object-detection pipeline.

---

## 12. Performance Requirements

Performance should be measured experimentally rather than assumed.

The following should be recorded:

- Inference time per person/frame
- Frames processed per second where applicable
- Memory usage
- Model size
- Classification accuracy
- Precision
- Recall
- F1-score

Real-time suitability depends on model size, image resolution, number of people per frame and available CPU/GPU hardware.

---

## 13. Security & Privacy Considerations

If the system is used with real camera footage:

- Access to recordings should be controlled.
- Prediction logs should be protected.
- Images should not be stored by default unless storage is explicitly enabled.
- The module should focus on activity classification rather than identification of individuals.

---

## 14. Testing Specification

| Testing Type | Requirement |
|---|---|
| Unit Testing | Validate preprocessing, model loading, prediction formatting and threshold logic. |
| Functional Testing | Verify image/video input and correct result generation. |
| Model Testing | Evaluate accuracy, precision, recall, F1-score and confusion matrix. |
| Robustness Testing | Test blur, lighting variation, different resolutions and partial crops where relevant. |
| Integration Testing | Pass person crops from the ANN detector and verify anomaly outputs. |
| Performance Testing | Measure latency and throughput on target hardware. |

---

## 15. Integration Contract With ANN Object Detection

The ANN object detector is responsible for **locating people**.

The Human Anomaly Detection module is responsible for **classifying the provided person image**.

This separation prevents duplicate detection logic and makes the anomaly component reusable.

### Integration Flow

```text
Camera / Video Frame
        |
        v
ANN Object Detector
        |
        v
Person Bounding Box
        |
        v
Crop Person Image
        |
        v
Human Anomaly Detector
        |
        v
Normal / Anomaly
        |
        v
Combined Result
```

### Integration Interface

```python
person_crop = frame[y1:y2, x1:x2]

result = anomaly_detector.predict(person_crop)

label = result["label"]
confidence = result["confidence"]
```

### Integration Responsibilities

| Component | Responsibility |
|---|---|
| ANN Object Detector | Detect people and generate bounding boxes. |
| Integration Layer | Crop the detected person and pass it to the anomaly detector. |
| Anomaly Detector | Classify the person crop. |
| Application | Display bounding box, label, confidence and optional alert. |

---

## 16. Acceptance Criteria

The project will be considered functional when:

- The project runs from the documented environment and requirements.
- The trained model can be loaded independently of the training script.
- A person image can be passed to the prediction interface and receives a valid result.
- Evaluation metrics are generated for the test set.
- The module can accept a crop produced by the future ANN object detector.
- No model-training code is required by the inference application.

---

## 17. Deployment Options

The module can initially be deployed as:

1. Local Python/OpenCV application.
2. Google Colab experimentation/training environment.
3. Flask/FastAPI inference API.
4. Integrated desktop/web application with the ANN object detector.

---

## 18. Future Technical Enhancements

- Temporal anomaly detection over sequences rather than individual frames.
- Multi-person tracking and per-person anomaly history.
- Model quantization or optimization for edge devices.
- Configurable alert and confidence policies.
- Centralized event logging.
- Dashboard integration.
- Multi-class anomaly recognition.
- CNN-LSTM or other temporal models for video-based anomaly detection.

---

## 19. Final Integration Goal

The final merged architecture is intended to be:

```text
                ANN OBJECT DETECTION PROJECT
                           |
                           v
                    Detect Objects
                           |
                           v
                     Detect Person
                           |
                           v
                    Person Crop
                           |
                           v
              HUMAN ANOMALY MODULE
                           |
                           v
                Anomaly Classification
                           |
                 +---------+---------+
                 |                   |
               Normal             Anomaly
                 |                   |
              Continue             Alert
```

The anomaly module therefore remains independent during development while having a clearly defined interface for the final ANN project.
