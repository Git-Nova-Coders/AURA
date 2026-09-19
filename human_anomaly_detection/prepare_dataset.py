import os
import shutil
import numpy as np
import cv2
import glob
import random

def setup_directories():
    base = 'dataset_split'
    splits = ['train', 'validation', 'test']
    classes = ['normal', 'anomaly']
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base, split, cls), exist_ok=True)
    return base

def main():
    base_dir = setup_directories()
    
    # Paths
    testing_frames_dir = os.path.join('dataset', 'shanghaitech', 'testing', 'frames')
    testing_masks_dir = os.path.join('dataset', 'shanghaitech', 'testing', 'test_frame_mask')
    
    sequences = [d for d in os.listdir(testing_frames_dir) if os.path.isdir(os.path.join(testing_frames_dir, d))]
    
    # To prevent leakage, split sequences into train/val/test
    random.seed(42)
    random.shuffle(sequences)
    
    num_seq = len(sequences)
    train_seqs = sequences[:int(0.6 * num_seq)]
    val_seqs = sequences[int(0.6 * num_seq):int(0.8 * num_seq)]
    test_seqs = sequences[int(0.8 * num_seq):]
    
    splits_dict = {
        'train': train_seqs,
        'validation': val_seqs,
        'test': test_seqs
    }
    
    # We will sample frames to not blow up disk space
    MAX_FRAMES_PER_CLASS_PER_SPLIT = 1500
    
    stats = {split: {'normal': 0, 'anomaly': 0} for split in splits_dict}
    
    for split, seqs in splits_dict.items():
        print(f"Processing {split} split...")
        normal_pool = []
        anomaly_pool = []
        
        for seq in seqs:
            mask_path = os.path.join(testing_masks_dir, f"{seq}.npy")
            if not os.path.exists(mask_path):
                continue
            
            mask = np.load(mask_path)
            frame_dir = os.path.join(testing_frames_dir, seq)
            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            
            # mask is 1D array of length equal to number of frames (usually)
            # 0 = normal, 1 = anomaly
            for i, frame_path in enumerate(frames):
                if i < len(mask):
                    if mask[i] == 1:
                        anomaly_pool.append(frame_path)
                    else:
                        normal_pool.append(frame_path)
        
        # Sample
        if len(normal_pool) > MAX_FRAMES_PER_CLASS_PER_SPLIT:
            normal_pool = random.sample(normal_pool, MAX_FRAMES_PER_CLASS_PER_SPLIT)
        if len(anomaly_pool) > MAX_FRAMES_PER_CLASS_PER_SPLIT:
            anomaly_pool = random.sample(anomaly_pool, MAX_FRAMES_PER_CLASS_PER_SPLIT)
            
        print(f"  {split} - Normal pool size: {len(normal_pool)}, Anomaly pool size: {len(anomaly_pool)}")
        
        # Copy files
        for i, src_path in enumerate(normal_pool):
            dst = os.path.join(base_dir, split, 'normal', f"norm_{split}_{i:05d}.jpg")
            shutil.copy2(src_path, dst)
            stats[split]['normal'] += 1
            
        for i, src_path in enumerate(anomaly_pool):
            dst = os.path.join(base_dir, split, 'anomaly', f"anom_{split}_{i:05d}.jpg")
            shutil.copy2(src_path, dst)
            stats[split]['anomaly'] += 1

    print("\nDataset Summary:")
    print("-" * 20)
    for split in splits_dict:
        print(f"{split.capitalize()}:")
        print(f"  Normal: {stats[split]['normal']}")
        print(f"  Anomaly: {stats[split]['anomaly']}")
    
    # Save the summary
    with open("dataset_summary.txt", "w") as f:
        f.write("Dataset Summary:\n")
        f.write("Dataset Name: ShanghaiTech (sampled subset for anomaly classification)\n")
        f.write("Source: Provided Zip File\n")
        f.write("Classes: Normal, Anomaly\n")
        f.write("Image/video format: JPEG frames extracted from video\n")
        f.write("Normal Definition: Regular pedestrian behavior, no unusual events.\n")
        f.write("Anomaly Definition: Unusual events like fighting, cycling on pedestrian paths, etc.\n\n")
        for split in splits_dict:
            f.write(f"{split.capitalize()}:\n")
            f.write(f"  Normal: {stats[split]['normal']}\n")
            f.write(f"  Anomaly: {stats[split]['anomaly']}\n")
            
if __name__ == "__main__":
    main()
