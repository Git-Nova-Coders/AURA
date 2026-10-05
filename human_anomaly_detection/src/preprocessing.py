import cv2
import numpy as np
import random
from . import config

def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Applies light augmentation to the image.
    Used only during training to prevent overfitting.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        
    # Random brightness variation
    if random.random() > 0.5:
        # Convert to HSV to adjust value
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Random brightness factor between 0.8 and 1.2
        factor = random.uniform(0.8, 1.2)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
    # Small rotation (-10 to 10 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        # Use BORDER_REPLICATE to fill empty space naturally
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
    return image

def preprocess_image(image: np.ndarray, is_training: bool = False) -> np.ndarray:
    """
    Preprocesses a raw BGR image (e.g. from cv2.imread or live camera) 
    into a model-ready format.
    
    This exact function MUST be used for both training and live inference.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for preprocessing.")

    # 1. Color Conversion: OpenCV uses BGR by default, neural networks expect RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Resize to target model dimensions
    img = cv2.resize(img, config.TARGET_SIZE)
    
    # 3. Apply Training Data Augmentations
    if is_training:
        img = augment_image(img)
        
    # 4. Pixel Normalization (Scale 0-255 down to 0.0-1.0)
    img_normalized = img.astype(np.float32) / 255.0
    
    return img_normalized
