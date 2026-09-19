import os
import cv2

def verify_dataset():
    base = 'dataset_split'
    splits = ['train', 'validation', 'test']
    classes = ['normal', 'anomaly']
    
    corrupt = 0
    empty = 0
    total = 0
    
    for split in splits:
        for cls in classes:
            d = os.path.join(base, split, cls)
            for f in os.listdir(d):
                p = os.path.join(d, f)
                total += 1
                img = cv2.imread(p)
                if img is None:
                    corrupt += 1
                elif img.size == 0:
                    empty += 1

    print(f"Total checked: {total}")
    print(f"Corrupt: {corrupt}")
    print(f"Empty: {empty}")

if __name__ == "__main__":
    verify_dataset()
