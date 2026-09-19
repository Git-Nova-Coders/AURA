import os
from src.data_loader import get_dataset_paths, data_generator

def test_pipeline():
    paths, labels = get_dataset_paths('train')
    if not paths:
        print("No training data found.")
        return
    
    print(f"Found {len(paths)} training samples.")
    
    gen = data_generator(paths, labels, batch_size=4, is_training=True, shuffle=True)
    
    batch_x, batch_y = next(gen)
    
    print(f"Batch X shape: {batch_x.shape}")
    print(f"Batch X dtype: {batch_x.dtype}")
    print(f"Batch X min: {batch_x.min()}, max: {batch_x.max()}")
    print(f"Batch Y shape: {batch_y.shape}")
    print(f"Batch Y: {batch_y}")
    print("Milestone 2 test passed successfully!")

if __name__ == "__main__":
    test_pipeline()
