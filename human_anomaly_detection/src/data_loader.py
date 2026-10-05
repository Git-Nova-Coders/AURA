import os
import cv2
import numpy as np
import random
from . import config
from .preprocessing import preprocess_image

def get_dataset_paths(split="train"):
    """
    Scans the dataset directory and returns a list of file paths and their labels.
    Label mapping: Normal = 0, Anomaly = 1.
    """
    base_dir = os.path.join(config.DATASET_DIR, split)
    
    paths = []
    labels = []
    
    # Label 0: Normal
    normal_dir = os.path.join(base_dir, 'normal')
    if os.path.exists(normal_dir):
        for f in os.listdir(normal_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(normal_dir, f))
                labels.append(0)
                
    # Label 1: Anomaly
    anomaly_dir = os.path.join(base_dir, 'anomaly')
    if os.path.exists(anomaly_dir):
        for f in os.listdir(anomaly_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(anomaly_dir, f))
                labels.append(1)
                
    return paths, labels

def data_generator(paths, labels, batch_size=None, is_training=False, shuffle=False):
    """
    A pure Python generator yielding (batch_images, batch_labels).
    This design is framework-agnostic. It can be wrapped into tf.data.Dataset 
    or PyTorch DataLoaders later during Milestone 3.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    num_samples = len(paths)
    indices = list(range(num_samples))
    
    while True:
        if shuffle:
            random.shuffle(indices)
            
        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:min(offset + batch_size, num_samples)]
            
            batch_images = []
            batch_labels = []
            
            for i in batch_indices:
                img_path = paths[i]
                label = labels[i]
                
                # cv2 reads images in BGR format, which is identical to the output format 
                # of a live camera cv2.VideoCapture() object.
                image = cv2.imread(img_path)
                if image is not None:
                    # Apply identical preprocessing as live inference
                    processed_image = preprocess_image(image, is_training=is_training)
                    batch_images.append(processed_image)
                    batch_labels.append(label)
                    
            if len(batch_images) > 0:
                yield np.array(batch_images), np.array(batch_labels)
                
        # Break out if we are not generating an infinite stream
        if not is_training and not shuffle:
            break
