"""
Image Dataset Loaders for 3NGIN3 Training

This module provides access to image datasets for testing the 3NGIN3 
neural reasoning modes and evaluation capabilities.

Supported datasets:
- CIFAR-10/100 - Multi-class image classification
- Synthetic geometric shapes for testing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_cifar10(subset_size: Optional[int] = 1000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the CIFAR-10 dataset.
    
    Args:
        subset_size: Number of samples to load (None for full dataset)
        
    Returns:
        X: Image data array (N, 32, 32, 3)
        y: Labels array
        metadata: Dataset information
    """
    logger.info("Loading CIFAR-10 dataset...")
    
    try:
        # Try using TensorFlow/Keras CIFAR-10
        try:
            import tensorflow as tf
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
            
            # Combine train and test
            X = np.concatenate([X_train, X_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0).flatten()
            
            # Normalize pixel values
            X = X.astype(np.float32) / 255.0
            
            if subset_size and subset_size < len(X):
                indices = np.random.choice(len(X), subset_size, replace=False)
                X = X[indices]
                y = y[indices]
                
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
            
            metadata = {
                'name': 'CIFAR-10',
                'task': 'multiclass_classification',
                'n_samples': len(X),
                'n_features': X.size // len(X),  # Total pixels per image
                'n_classes': 10,
                'image_shape': (32, 32, 3),
                'class_names': class_names,
                'description': 'CIFAR-10 image classification dataset'
            }
            
            logger.info(f"Loaded CIFAR-10: {metadata['n_samples']} samples, shape {X.shape}")
            return X, y, metadata
            
        except ImportError:
            logger.warning("TensorFlow not available, trying torchvision...")
            
            # Try using torchvision
            import torchvision
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader
            
            transform = transforms.Compose([transforms.ToTensor()])
            
            # Download and load train set
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)
            # Download and load test set  
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform)
            
            # Convert to numpy
            train_data = []
            train_labels = []
            for data, label in trainset:
                train_data.append(data.numpy().transpose(1, 2, 0))  # CHW to HWC
                train_labels.append(label)
                
            test_data = []
            test_labels = []
            for data, label in testset:
                test_data.append(data.numpy().transpose(1, 2, 0))  # CHW to HWC
                test_labels.append(label)
            
            X = np.array(train_data + test_data)
            y = np.array(train_labels + test_labels)
            
            if subset_size and subset_size < len(X):
                indices = np.random.choice(len(X), subset_size, replace=False)
                X = X[indices]
                y = y[indices]
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
            
            metadata = {
                'name': 'CIFAR-10',
                'task': 'multiclass_classification',
                'n_samples': len(X),
                'n_features': X.size // len(X),
                'n_classes': 10,
                'image_shape': (32, 32, 3),
                'class_names': class_names,
                'description': 'CIFAR-10 image classification dataset'
            }
            
            logger.info(f"Loaded CIFAR-10: {metadata['n_samples']} samples, shape {X.shape}")
            return X, y, metadata
            
    except Exception as e:
        logger.warning(f"Failed to load real CIFAR-10: {e}")
        logger.info("Generating synthetic image-like dataset...")
        
    # Generate synthetic image data as fallback
    np.random.seed(42)
    n_samples = subset_size if subset_size else 1000
    
    # Generate synthetic 32x32 RGB images
    X = np.random.rand(n_samples, 32, 32, 3).astype(np.float32)
    
    # Create synthetic patterns for different classes
    y = np.random.randint(0, 10, n_samples)
    
    # Add some class-specific patterns
    for i in range(n_samples):
        class_label = y[i]
        # Add class-specific color bias
        color_channel = class_label % 3
        X[i, :, :, color_channel] += 0.3
        
        # Add simple geometric patterns
        if class_label < 5:
            # Add horizontal lines for classes 0-4
            X[i, 10:12, :, :] = 0.8
        else:
            # Add vertical lines for classes 5-9
            X[i, :, 15:17, :] = 0.8
    
    # Ensure values are in [0, 1]
    X = np.clip(X, 0, 1)
    
    class_names = ['synth_0', 'synth_1', 'synth_2', 'synth_3', 'synth_4',
                   'synth_5', 'synth_6', 'synth_7', 'synth_8', 'synth_9']
    
    metadata = {
        'name': 'CIFAR-10_Synthetic',
        'task': 'multiclass_classification',
        'n_samples': len(X),
        'n_features': X.size // len(X),
        'n_classes': 10,
        'image_shape': (32, 32, 3),
        'class_names': class_names,
        'description': 'Synthetic CIFAR-10-like image dataset'
    }
    
    logger.info(f"Generated synthetic CIFAR-10: {metadata['n_samples']} samples, shape {X.shape}")
    return X, y, metadata

def load_geometric_shapes(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate simple geometric shapes for testing neural reasoning.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        X: Image data array (N, 28, 28, 1) 
        y: Shape labels (0=circle, 1=square, 2=triangle)
        metadata: Dataset information
    """
    logger.info(f"Generating {n_samples} geometric shapes...")
    
    try:
        from PIL import Image, ImageDraw
        
        X = []
        y = []
        
        for i in range(n_samples):
            # Create 28x28 grayscale image
            img = Image.new('L', (28, 28), 0)  # Black background
            draw = ImageDraw.Draw(img)
            
            shape_type = i % 3
            y.append(shape_type)
            
            # Add some randomness to position and size
            center_x = 14 + np.random.randint(-3, 4)
            center_y = 14 + np.random.randint(-3, 4)
            size = 8 + np.random.randint(-2, 3)
            
            if shape_type == 0:  # Circle
                draw.ellipse([center_x-size, center_y-size, 
                             center_x+size, center_y+size], fill=255)
            elif shape_type == 1:  # Square
                draw.rectangle([center_x-size, center_y-size,
                               center_x+size, center_y+size], fill=255)
            else:  # Triangle
                points = [(center_x, center_y-size),
                         (center_x-size, center_y+size),
                         (center_x+size, center_y+size)]
                draw.polygon(points, fill=255)
            
            # Add noise
            img_array = np.array(img, dtype=np.float32) / 255.0
            noise = np.random.normal(0, 0.1, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            
            X.append(img_array)
        
        X = np.array(X)
        X = X.reshape(n_samples, 28, 28, 1)  # Add channel dimension
        y = np.array(y)
        
    except ImportError:
        logger.warning("PIL not available, generating simple synthetic shapes...")
        
        # Simple synthetic shapes without PIL
        X = np.zeros((n_samples, 28, 28, 1))
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            shape_type = i % 3
            y[i] = shape_type
            
            center = 14
            size = 6
            
            if shape_type == 0:  # Circle (rough approximation)
                for x in range(28):
                    for y_coord in range(28):
                        if (x - center)**2 + (y_coord - center)**2 <= size**2:
                            X[i, x, y_coord, 0] = 1.0
            elif shape_type == 1:  # Square
                X[i, center-size:center+size, center-size:center+size, 0] = 1.0
            else:  # Triangle (rough approximation)
                for x in range(center-size, center+size):
                    for y_coord in range(center, center+size):
                        if x - (center-size) <= y_coord - center:
                            X[i, x, y_coord, 0] = 1.0
        
        # Add noise
        noise = np.random.normal(0, 0.1, X.shape)
        X = np.clip(X + noise, 0, 1)
    
    metadata = {
        'name': 'Geometric_Shapes',
        'task': 'multiclass_classification',
        'n_samples': len(X),
        'n_features': X.size // len(X),
        'n_classes': 3,
        'image_shape': (28, 28, 1),
        'class_names': ['circle', 'square', 'triangle'],
        'description': 'Simple geometric shapes for pattern recognition testing'
    }
    
    logger.info(f"Generated geometric shapes: {metadata['n_samples']} samples, shape {X.shape}")
    return X, y, metadata

def get_all_image_datasets(subset_size: int = 1000) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Load all available image datasets.
    
    Args:
        subset_size: Number of samples for CIFAR-10
        
    Returns:
        Dictionary mapping dataset names to (X, y, metadata) tuples
    """
    datasets_dict = {}
    
    dataset_loaders = {
        'cifar10': lambda: load_cifar10(subset_size),
        'geometric_shapes': lambda: load_geometric_shapes(subset_size)
    }
    
    for name, loader in dataset_loaders.items():
        try:
            X, y, metadata = loader()
            datasets_dict[name] = (X, y, metadata)
            logger.info(f"Successfully loaded {name}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    
    return datasets_dict

if __name__ == "__main__":
    # Test loading all datasets
    logging.basicConfig(level=logging.INFO)
    datasets_dict = get_all_image_datasets(subset_size=100)  # Small subset for testing
    
    print("\n=== IMAGE DATASETS SUMMARY ===")
    for name, (X, y, metadata) in datasets_dict.items():
        print(f"\n{metadata['name']}:")
        print(f"  Task: {metadata['task']}")
        print(f"  Samples: {metadata['n_samples']}")
        print(f"  Image shape: {metadata['image_shape']}")
        print(f"  Classes: {metadata['n_classes']}")
        print(f"  Class names: {metadata['class_names'][:5]}...")  # Show first 5
        print(f"  Description: {metadata['description']}")
        print(f"  Data shape: {X.shape}")
        print(f"  Label range: {y.min()} - {y.max()}")