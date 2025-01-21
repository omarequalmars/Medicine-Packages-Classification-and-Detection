import h5py
import numpy as np
from PIL import Image
import os

def create_dataset_h5(dataset_path, output_file, target_size=(224, 224)):
    """
    Creates H5 dataset with resized images to ensure uniform dimensions
    
    Parameters:
        dataset_path: Path to dataset folder
        output_file: Output H5 file path
        target_size: Tuple of (height, width) for resizing
    """
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    with h5py.File(output_file, 'w') as h5f:
        # Initialize empty lists
        images_list = []
        labels_list = []
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            class_idx = class_to_idx[class_name]
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Resize image to target size
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        # Convert to numpy array
                        img_array = np.array(img)
                        images_list.append(img_array)
                        labels_list.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        
        # Convert lists to arrays with uniform shape
        images = np.stack(images_list, axis=0)
        labels = np.array(labels_list)
        
        # Create datasets
        h5f.create_dataset('images', data=images, compression='gzip', compression_opts=9)
        h5f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
        h5f.attrs['class_names'] = np.array(classes, dtype='S')

def create_splits(h5_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Creates separate train/validation/test H5 files from the input H5 file
    
    Parameters:
        h5_file (str): Path to input H5 file
        output_dir (str): Directory to save split files
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data 
        test_ratio (float): Ratio of test data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file, 'r') as src:
        n_samples = len(src['labels'])
        indices = np.random.permutation(n_samples)
        
        # Calculate split sizes
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        
        # Create split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size] 
        test_indices = indices[train_size+val_size:]
        
        # Create train file
        with h5py.File(os.path.join(output_dir, 'train.h5'), 'w') as f_train:
            for key in src.keys():
                data = src[key][:]
                f_train.create_dataset(key, data=data[train_indices])
            # Copy attributes
            for key, value in src.attrs.items():
                f_train.attrs[key] = value
                
        # Create validation file
        with h5py.File(os.path.join(output_dir, 'val.h5'), 'w') as f_val:
            for key in src.keys():
                data = src[key][:]
                f_val.create_dataset(key, data=data[val_indices])
            # Copy attributes
            for key, value in src.attrs.items():
                f_val.attrs[key] = value
                
        # Create test file
        with h5py.File(os.path.join(output_dir, 'test.h5'), 'w') as f_test:
            for key in src.keys():
                data = src[key][:]
                f_test.create_dataset(key, data=data[test_indices])
            # Copy attributes  
            for key, value in src.attrs.items():
                f_test.attrs[key] = value

# Usage
if __name__ == "__main__":
    dataset_path = "Original Dataset"  # Change this to your dataset path
    output_file = "medications_dataset.h5"
    create_dataset_h5(dataset_path, output_file)
    create_splits(output_file, "splits")
