# Dataset Validation Report (Milestone 1)

## 1. Dataset Information
- **Dataset name:** ShanghaiTech Campus Dataset
- **Source:** Provided zip file containing `shanghaitech.tar.gz` chunks, extracted locally.
- **Classes:** Normal (0), Anomaly (1)
- **Image/video format:** JPEG frames extracted from standard RGB video sequences.
- **Licensing/usage considerations:** Academic and research use. Must cite the original ShanghaiTech dataset paper.

## 2. Definitions
- **Normal:** Regular campus activities, such as people walking, talking, or carrying normal items.
- **Anomaly:** Unusual events, such as fighting, pushing, riding a bicycle on a pedestrian path, or driving a vehicle in a restricted area.

## 3. Train/Validation/Test Split Strategy
To avoid data leakage, frames from the exact same sequence are strictly kept within the same split. 
For binary classification purposes (as suggested in Milestone 3), anomaly frames have been included in the training set by re-purposing a portion of the original testing sequences (which contained both normal and anomaly frames). The testing sequences were shuffled and split randomly (60% Train, 20% Validation, 20% Test).
A maximum of 1500 frames per class per split were sampled to construct a balanced and manageable dataset for baseline model development.

## 4. Dataset Summary Numbers
*All numbers reflect the processed, sampled dataset (from actual execution):*

Train:
Normal: 1500
Anomaly: 1500

Validation:
Normal: 1500
Anomaly: 1500

Test:
Normal: 1500
Anomaly: 1500

## 5. Sanity Checks
- **Corrupt files:** 0 detected.
- **Empty images:** 0 detected.
- **Incorrect labels:** Frames extracted using the provided `.npy` mask files. Assumed accurate from dataset source.
- **Severe class imbalance:** Handled by explicit sampling limits (1500 per class per split).
- **Duplicate or near-duplicate samples:** Consecutive video frames inherently contain near-duplicates, but no cross-split leakage exists due to video-level separation.
- **Data leakage:** Verified that train, validation, and test splits utilize strictly separate sequence IDs.
