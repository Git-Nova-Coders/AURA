# AI Agent Development Guide
# Human Anomaly Detection Using Computer Vision

**Version:** 1.0  
**Purpose:** Step-by-step development specification for an AI coding agent.  
**Project Status:** Standalone module; integration with the ANN Object Detection project will happen later.

---

## 1. Agent Mission

Develop a modular **Human Anomaly Detection Using Computer Vision** project that can:

1. Accept images, video files, or camera frames.
2. Identify/crop human regions when a standalone human detector is used.
3. Classify the human region as **Normal** or **Anomaly** according to the selected dataset definition.
4. Produce a prediction label and confidence score.
5. Train, validate, test, save and reload the ML model.
6. Provide a clean inference interface that can later receive person crops from the user's ANN object-detection project.
7. Remain independent from the ANN project until the explicit integration milestone.

**Important:** Do not merge this project into the ANN object-detection project during the initial development milestones.

---

# 2. Core Development Principles

The AI agent must follow these principles throughout development:

### 2.1 Modular Design

Keep these responsibilities separate:

- Dataset handling
- Preprocessing
- Model definition
- Training
- Evaluation
- Inference
- Visualization
- Application/UI
- Configuration

Do not place the entire project inside one Python file.

### 2.2 Integration-Ready Design

The anomaly detector must eventually support:

```python
result = anomaly_detector.predict(person_image)
```

The anomaly model must not depend on how the person image was obtained.

### 2.3 Reproducibility

The agent must:

- Use a `requirements.txt`.
- Keep configuration values centralized.
- Save the trained model.
- Record important training parameters.
- Use reproducible train/validation/test splitting where possible.

### 2.4 Avoid Premature Complexity

Start with the simplest reliable architecture.

Do not introduce:

- Complex tracking
- Real-time multi-camera systems
- Cloud deployment
- CNN-LSTM
- Transformers
- Large-scale optimization

until the baseline system is working.

### 2.5 No Fabricated Results

The agent must never invent:

- Accuracy
- Precision
- Recall
- F1-score
- Dataset size
- Training time
- FPS
- Confusion matrix values

All results must come from actual execution.

---

# 3. Recommended Project Structure

The agent should work toward this structure:

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
│   ├── anomaly_model.keras
│   └── model_config.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── anomaly_detector.py
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── predictions/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model_loading.py
│   └── test_prediction.py
│
├── app.py
├── requirements.txt
├── README.md
├── PRT.md
├── SSD.md
└── AI_AGENT_DEVELOPMENT.md
```

The agent may adjust the structure when there is a strong technical reason, but must explain the change before making it.

---

# 4. Development Milestones

## Milestone 0 — Project Initialization

### Goal

Create a clean and runnable project skeleton.

### Tasks

- Create the project directory.
- Create the required subdirectories.
- Create a Python virtual environment if working locally.
- Create `requirements.txt`.
- Create `README.md`.
- Create placeholder Python modules.
- Create `.gitignore`.
- Create configuration handling.

### Expected Output

The project should have:

```text
src/
models/
results/
tests/
dataset/
```

and the basic Python files.

### Completion Criteria

- Project structure exists.
- Python files import without errors.
- Dependencies are documented.
- README contains setup instructions.

### Do Not Proceed If

- Imports are broken.
- Folder structure is inconsistent.
- Dataset assumptions have not been documented.

---

# Milestone 1 — Dataset Selection and Validation

### Goal

Obtain and validate a suitable dataset.

### Tasks

1. Select a dataset appropriate for human anomaly/activity detection.
2. Document:
   - Dataset name
   - Source
   - Classes
   - Number of samples
   - Image/video format
   - Licensing/usage considerations
3. Define what **Normal** and **Anomaly** mean for this project.
4. Organize the dataset into train/validation/test sets.
5. Check for:
   - Corrupt files
   - Empty images
   - Incorrect labels
   - Severe class imbalance
   - Duplicate or near-duplicate samples
   - Data leakage

### Important Rule

If the dataset is video-based, avoid placing frames from the same original video into different splits when this could cause leakage.

### Expected Output

A validated dataset with documented classes and split strategy.

### Completion Criteria

The agent can produce a dataset summary such as:

```text
Train:
Normal: XXXX
Anomaly: XXXX

Validation:
Normal: XXXX
Anomaly: XXXX

Test:
Normal: XXXX
Anomaly: XXXX
```

All numbers must be obtained from the actual dataset.

---

# Milestone 2 — Data Preprocessing Pipeline

### Goal

Build a reusable preprocessing pipeline.

### Tasks

Implement:

- Image loading
- Resize
- Color conversion where required
- Pixel normalization
- Label encoding
- Batch generation

Optional training augmentation:

- Horizontal flip where appropriate
- Small rotation
- Zoom
- Translation
- Brightness variation

### Important Rules

- Apply augmentation only to training data.
- Keep validation and test preprocessing deterministic.
- Use the exact same inference preprocessing used during training.
- Do not distort images unnecessarily.

### Expected Interface

```python
image = preprocess_image(image)
```

### Completion Criteria

The agent can load an image and return a model-ready tensor with the correct shape and numerical range.

---

# Milestone 3 — Baseline ML Model

### Goal

Build the first working anomaly classifier.

### Preferred Initial Approach

Use a CNN image classifier.

Possible baseline:

```text
Input
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPooling
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPooling
  ↓
Flatten / GlobalAveragePooling
  ↓
Dense
  ↓
Dropout
  ↓
Output
```

For binary classification, a sigmoid output can be used.

### Tasks

- Create `model.py`.
- Define model architecture.
- Print model summary.
- Define loss function.
- Define optimizer.
- Define evaluation metrics.
- Add regularization if needed.

### Completion Criteria

The model can:

- Build successfully.
- Accept a sample batch.
- Produce predictions.
- Complete at least one training epoch without errors.

---

# Milestone 4 — Training Pipeline

### Goal

Create a complete reproducible training pipeline.

### Tasks

Implement:

```bash
python -m src.train
```

The training pipeline should:

1. Load configuration.
2. Load dataset.
3. Preprocess data.
4. Create model.
5. Train model.
6. Validate model.
7. Save best model.
8. Save training history.
9. Print final training information.

### Recommended Training Controls

Consider:

- Early stopping
- Model checkpointing
- Learning-rate scheduling

Do not add every callback automatically; use only those that improve reliability.

### Expected Output

```text
models/
└── anomaly_model.keras

results/
└── training_history.png
```

---

# Milestone 5 — Evaluation

### Goal

Evaluate the model objectively on unseen test data.

### Required Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Where meaningful, also calculate:

- ROC-AUC
- Specificity
- Sensitivity

### Tasks

Create:

```bash
python -m src.evaluate
```

Generate:

```text
results/
├── confusion_matrix.png
└── classification_report.txt
```

### Important Interpretation Rule

The agent must not report accuracy alone.

For anomaly detection, examine false positives and false negatives separately.

### Completion Criteria

Evaluation is performed on the held-out test set and all reported metrics come from actual execution.

---

# Milestone 6 — Inference Engine

### Goal

Create a clean reusable prediction interface.

### Required Interface

```python
from src.anomaly_detector import AnomalyDetector

detector = AnomalyDetector("models/anomaly_model.keras")

result = detector.predict(person_image)
```

Expected structure:

```python
{
    "label": "Normal",
    "confidence": 0.94
}
```

### Tasks

- Load model once.
- Apply correct preprocessing.
- Generate prediction.
- Convert prediction to label.
- Return confidence.
- Handle invalid input gracefully.

### Completion Criteria

A single image/person crop can be passed to the detector without running the training pipeline.

---

# Milestone 7 — Image Inference

### Goal

Allow users to test the model using individual images.

### Tasks

Support:

```bash
python -m src.predict --image path/to/image.jpg
```

Display:

```text
Prediction: Anomaly
Confidence: 91.2%
```

Optionally save an annotated result.

### Completion Criteria

The system can successfully classify previously unseen images.

---

# Milestone 8 — Video / OpenCV Inference

### Goal

Run the classifier on video frames.

### Tasks

- Read video using OpenCV.
- Process frames.
- Run prediction.
- Display label and confidence.
- Allow the user to exit safely.
- Avoid unnecessary model reloading on every frame.

### Important Performance Rule

Do not run expensive initialization inside the frame loop.

Bad:

```python
while True:
    model = load_model(...)
```

Good:

```python
model = load_model(...)

while True:
    ...
```

### Completion Criteria

The model can process a video without crashing and produces visible predictions.

---

# Milestone 9 — Standalone Human Detection

### Goal

Make the standalone project capable of receiving a complete frame and obtaining a person region when necessary.

### Important Architecture

```text
Frame
  ↓
Human/Person Detector
  ↓
Person Crop
  ↓
Anomaly Detector
  ↓
Normal / Anomaly
```

### Important Rule

The person detector and anomaly classifier must remain separate components.

This allows the future ANN object detector to replace the standalone person detector.

### Completion Criteria

A complete frame can be processed through:

```text
Frame → Person Detection → Crop → Anomaly Classification
```

where technically appropriate for the selected dataset.

---

# Milestone 10 — Application Layer

### Goal

Create a simple user-facing application.

The application may support:

- Image upload
- Video input
- Camera input
- Prediction display
- Confidence display

A simple OpenCV interface is sufficient initially.

A Flask/FastAPI interface may be added later.

### Do Not

Do not spend significant time creating a complex frontend before the ML pipeline is stable.

---

# Milestone 11 — Testing

### Goal

Verify that the project is reliable.

### Unit Tests

Test:

- Preprocessing
- Model loading
- Prediction output
- Invalid image handling
- Configuration loading

### Integration Tests

Test:

```text
Input frame
   ↓
Person crop
   ↓
Anomaly detector
   ↓
Prediction
```

### Model Tests

Test:

- Normal samples
- Anomaly samples
- Different resolutions
- Lighting variation
- Blur where relevant
- Partial crops where relevant

### Completion Criteria

Tests pass and failures are documented rather than hidden.

---

# Milestone 12 — Model Improvement

### Goal

Improve the baseline only after measuring its weaknesses.

### Analyze

- Training vs validation curves
- Confusion matrix
- False positives
- False negatives
- Class imbalance
- Overfitting
- Underfitting

### Possible Improvements

Depending on actual results:

- Data augmentation
- Class weights
- Dropout
- Batch normalization
- Transfer learning
- Learning-rate tuning
- Better data cleaning
- More representative training samples

### Important Rule

Do not change multiple major factors simultaneously without recording the experiment.

---

# Milestone 13 — Experiment Tracking

### Goal

Make model improvements reproducible.

Record:

```text
Experiment ID
Dataset version
Model architecture
Image size
Batch size
Epochs
Optimizer
Learning rate
Augmentation
Training metrics
Validation metrics
Test metrics
Observations
```

Example:

```text
Experiment: EXP-003
Model: CNN-Baseline-v2
Image size: 224x224
Batch size: 32
Epochs: 30
Optimizer: Adam
Learning rate: 0.001

Observation:
Validation performance improved after adding dropout.
```

---

# Milestone 14 — Integration Preparation

### Goal

Prepare the anomaly module for merging with the ANN Object Detection project.

### Required Interface

The anomaly module must accept:

```python
person_image
```

and return:

```python
{
    "label": "...",
    "confidence": ...
}
```

### Integration Architecture

```text
                 ANN OBJECT DETECTOR
                         |
                         v
                 Detect Person
                         |
                         v
                Bounding Box
                         |
                         v
                    Crop Image
                         |
                         v
             HUMAN ANOMALY MODULE
                         |
                         v
             AnomalyDetector.predict()
                         |
                 +-------+-------+
                 |               |
               Normal          Anomaly
                 |               |
               Display          Alert
```

### Important Rule

The anomaly module must not assume:

- Which object detector is being used.
- Which ANN architecture is being used.
- Which frontend is being used.
- Which camera source is being used.

It should only require the person image/crop.

---

# Milestone 15 — Final Integration With ANN Project

### Goal

Merge the completed anomaly module into the existing ANN object-detection project.

### Before Integration

Create a backup or Git branch.

Suggested:

```bash
git checkout -b anomaly-integration
```

### Integration Steps

1. Copy/import the anomaly module.
2. Preserve its internal modular structure.
3. Connect the ANN detector's person bounding box to the anomaly module.
4. Crop each detected person.
5. Send each crop to `AnomalyDetector.predict()`.
6. Add anomaly label/confidence to the object detector's output.
7. Display both:
   - Object/person detection
   - Anomaly status
8. Test multiple people in one frame.
9. Test cases where no person is detected.
10. Test cases where the anomaly model fails or returns an invalid result.

### Final Combined Flow

```text
Camera / Video
      ↓
ANN Object Detector
      ↓
Detect Person(s)
      ↓
Bounding Box(es)
      ↓
Crop Person(s)
      ↓
Human Anomaly Detector
      ↓
Normal / Anomaly
      ↓
Combined Visualization
```

---

# 5. Agent Coding Rules

## Rule 1 — Inspect Before Editing

Before modifying existing code:

1. Inspect the file.
2. Understand its current purpose.
3. Identify dependencies.
4. Make the smallest necessary change.

Do not overwrite working code without reason.

---

## Rule 2 — Preserve Working Components

When integrating new features, existing working functionality must continue to operate.

If a change may break an existing component, explain the risk and isolate the change.

---

## Rule 3 — One Milestone at a Time

The agent should not implement all milestones in one uncontrolled operation.

For each milestone:

1. Implement.
2. Run.
3. Test.
4. Fix errors.
5. Verify output.
6. Document completion.
7. Proceed to the next milestone.

---

## Rule 4 — Ask Before Major Architectural Changes

The agent should ask the developer before:

- Replacing the primary ML framework.
- Changing the dataset definition.
- Replacing the entire model architecture.
- Merging projects prematurely.
- Removing major project components.
- Introducing a large external dependency.

---

## Rule 5 — Use Relative Paths

Avoid hard-coded paths such as:

```python
C:/Users/Name/Desktop/project/dataset
```

Prefer:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
```

---

## Rule 6 — Configuration Must Be Centralized

Important settings should be stored in one location.

Example:

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
ANOMALY_THRESHOLD = 0.5
MODEL_PATH = "models/anomaly_model.keras"
```

Values should be changed through configuration rather than scattered throughout the code.

---

## Rule 7 — Handle Errors Clearly

Use meaningful error messages.

Example:

```text
Error: Model file not found at models/anomaly_model.keras
Please train the model or provide a valid model path.
```

Avoid silent failures.

---

# 6. Git / Version Control Milestones

Recommended commit points:

```text
1. Initialize human anomaly detection project
2. Add dataset pipeline
3. Add preprocessing module
4. Add baseline CNN model
5. Add training pipeline
6. Add evaluation pipeline
7. Add anomaly detector inference class
8. Add image inference
9. Add video inference
10. Add standalone person detection
11. Add tests
12. Improve baseline model
13. Prepare ANN integration interface
14. Integrate anomaly detection with ANN object detector
```

Use clear commit messages and avoid one huge commit containing the entire project.

---

# 7. Definition of Done

The standalone project is complete only when all of the following are true:

- [ ] Dataset is documented.
- [ ] Dataset is split correctly.
- [ ] Preprocessing is implemented.
- [ ] Baseline model is trained.
- [ ] Model is saved.
- [ ] Model can be reloaded.
- [ ] Evaluation metrics are generated.
- [ ] Confusion matrix is generated.
- [ ] Image inference works.
- [ ] Video inference works where supported.
- [ ] Error handling exists.
- [ ] Tests exist and pass.
- [ ] README contains setup and usage instructions.
- [ ] PRT and SSD are included.
- [ ] Anomaly detector has a reusable `predict()` interface.
- [ ] Integration with the ANN detector has been tested.

---

# 8. Final Integration Definition of Done

After merging with the ANN project:

- [ ] ANN object detection still works.
- [ ] Person bounding boxes are correctly generated.
- [ ] Person crops are correctly extracted.
- [ ] Crops are passed to the anomaly detector.
- [ ] Anomaly labels are returned correctly.
- [ ] Confidence values are displayed correctly.
- [ ] Multiple people can be processed if supported.
- [ ] No-person frames do not crash the application.
- [ ] Model loading happens only once where appropriate.
- [ ] Existing ANN functionality is preserved.
- [ ] Combined README/documentation is updated.

---

# 9. Recommended Development Order

The AI agent should follow this exact high-level order:

```text
M0  Project Setup
 ↓
M1  Dataset
 ↓
M2  Preprocessing
 ↓
M3  Baseline Model
 ↓
M4  Training
 ↓
M5  Evaluation
 ↓
M6  Inference Engine
 ↓
M7  Image Testing
 ↓
M8  Video/OpenCV Testing
 ↓
M9  Standalone Human Detection
 ↓
M10 Application
 ↓
M11 Testing
 ↓
M12 Model Improvement
 ↓
M13 Experiment Tracking
 ↓
M14 Integration Preparation
 ↓
M15 ANN Integration
```

---

# 10. Agent Progress Report Format

After completing each milestone, the AI agent should report:

```text
MILESTONE: M__

STATUS: COMPLETE / PARTIAL / BLOCKED

Completed:
- ...
- ...

Files Created:
- ...
- ...

Files Modified:
- ...
- ...

Tests Run:
- ...
- ...

Results:
- ...

Issues:
- ...

Next Milestone:
- ...
```

If a milestone is blocked, the agent must explain exactly what is blocking it and what information/action is required.

---

# 11. Important Project Constraint

This project is being developed as a **standalone Human Anomaly Detection module first**.

The later ANN Object Detection project is the **upstream detection component**.

Therefore:

```text
Current Project:

Frame
  ↓
Person Detection
  ↓
Anomaly Detection


Future Combined Project:

Frame
  ↓
ANN Object Detection
  ↓
Person Bounding Box
  ↓
Human Anomaly Detection
```

The standalone person detector is replaceable.

The anomaly classifier is reusable.

This separation is the most important architectural requirement for future merging.

---

# 12. Final Expected Architecture

```text
human_anomaly_detection/
│
├── dataset/
│
├── models/
│   └── anomaly_model.keras
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── anomaly_detector.py
│
├── results/
│
├── tests/
│
├── app.py
├── requirements.txt
├── README.md
├── PRT.md
├── SSD.md
└── AI_AGENT_DEVELOPMENT.md
```

**End of AI Agent Development Guide**
