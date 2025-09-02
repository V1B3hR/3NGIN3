"""
UCI Machine Learning Repository Dataset Loaders for 3NGIN3 Training

This module provides easy access to UCI datasets for testing the 3NGIN3 
reasoning modes and evaluation capabilities.

Supported datasets:
- Adult (Census Income) - Binary classification
- Heart Disease - Binary classification  
- Covertype (Forest Cover Type) - Multi-class classification
- Wine Quality - Regression/Classification
- Diabetes - Regression
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
import io
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_covertype() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the Covertype (Forest Cover Type) dataset from UCI.
    
    Returns:
        X: Features DataFrame
        y: Target Series
        metadata: Dataset information
    """
    logger.info("Loading Covertype dataset...")
    data = datasets.fetch_covtype()
    
    feature_names = [f'feature_{i}' for i in range(data.data.shape[1])]
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name='cover_type')
    
    metadata = {
        'name': 'Covertype',
        'task': 'multiclass_classification', 
        'n_samples': len(X),
        'n_features': len(X.columns),
        'n_classes': len(np.unique(y)),
        'description': 'Forest cover type prediction from cartographic variables'
    }
    
    logger.info(f"Loaded Covertype: {metadata['n_samples']} samples, {metadata['n_features']} features, {metadata['n_classes']} classes")
    return X, y, metadata

def load_diabetes_uci() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the Diabetes dataset from UCI.
    
    Returns:
        X: Features DataFrame
        y: Target Series  
        metadata: Dataset information
    """
    logger.info("Loading Diabetes dataset...")
    data = datasets.load_diabetes()
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='diabetes_progression')
    
    metadata = {
        'name': 'Diabetes',
        'task': 'regression',
        'n_samples': len(X),
        'n_features': len(X.columns),
        'description': 'Diabetes disease progression prediction'
    }
    
    logger.info(f"Loaded Diabetes: {metadata['n_samples']} samples, {metadata['n_features']} features")
    return X, y, metadata

def load_wine_quality() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the Wine Quality dataset from UCI.
    
    Returns:
        X: Features DataFrame
        y: Target Series
        metadata: Dataset information
    """
    logger.info("Loading Wine Quality dataset...")
    data = datasets.load_wine()
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='wine_class')
    
    metadata = {
        'name': 'Wine_Quality',
        'task': 'multiclass_classification',
        'n_samples': len(X),
        'n_features': len(X.columns),
        'n_classes': len(np.unique(y)),
        'description': 'Wine quality classification from chemical analysis'
    }
    
    logger.info(f"Loaded Wine Quality: {metadata['n_samples']} samples, {metadata['n_features']} features, {metadata['n_classes']} classes")
    return X, y, metadata

def load_adult_census() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the Adult (Census Income) dataset from UCI.
    
    Returns:
        X: Features DataFrame
        y: Target Series
        metadata: Dataset information
    """
    logger.info("Loading Adult Census dataset...")
    
    try:
        # Try to download Adult dataset from UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                       'marital_status', 'occupation', 'relationship', 'race', 'sex',
                       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
        
        data = pd.read_csv(io.StringIO(response.text), names=column_names, skipinitialspace=True)
        
        # Clean data
        data = data.replace('?', np.nan).dropna()
        
        # Encode categorical variables
        categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                           'relationship', 'race', 'sex', 'native_country']
        
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        # Separate features and target
        X = data.drop('income', axis=1)
        y = data['income'].apply(lambda x: 1 if '>50K' in x else 0)
        y.name = 'high_income'
        
        metadata = {
            'name': 'Adult_Census',
            'task': 'binary_classification',
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_classes': 2,
            'description': 'Predict whether income exceeds $50K/yr based on census data'
        }
        
        logger.info(f"Loaded Adult Census: {metadata['n_samples']} samples, {metadata['n_features']} features")
        return X, y, metadata
        
    except Exception as e:
        logger.warning(f"Failed to load Adult dataset from UCI: {e}")
        logger.info("Using synthetic adult-like dataset instead...")
        
        # Generate synthetic adult-like data as fallback
        np.random.seed(42)
        n_samples = 32561  # Approximate size of real dataset
        
        X = pd.DataFrame({
            'age': np.random.randint(17, 90, n_samples),
            'education_num': np.random.randint(1, 16, n_samples),
            'hours_per_week': np.random.randint(1, 99, n_samples),
            'capital_gain': np.random.exponential(100, n_samples),
            'capital_loss': np.random.exponential(50, n_samples)
        })
        
        # Simple income prediction based on features
        income_score = (X['age'] * 0.01 + X['education_num'] * 0.1 + 
                       X['hours_per_week'] * 0.02 + X['capital_gain'] * 0.0001)
        y = pd.Series((income_score > np.percentile(income_score, 75)).astype(int), name='high_income')
        
        metadata = {
            'name': 'Adult_Census_Synthetic',
            'task': 'binary_classification',
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_classes': 2,
            'description': 'Synthetic adult census-like data for income prediction'
        }
        
        logger.info(f"Generated synthetic Adult Census: {metadata['n_samples']} samples, {metadata['n_features']} features")
        return X, y, metadata

def load_heart_disease() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the Heart Disease dataset from UCI.
    
    Returns:
        X: Features DataFrame
        y: Target Series
        metadata: Dataset information
    """
    logger.info("Loading Heart Disease dataset...")
    
    try:
        # Try to download Heart Disease dataset from UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        data = pd.read_csv(io.StringIO(response.text), names=column_names)
        
        # Clean data - replace missing values marked as '?'
        data = data.replace('?', np.nan)
        data = data.dropna()
        
        # Convert to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna()  # Remove any remaining non-numeric rows
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Convert to binary
        y.name = 'heart_disease'
        
        metadata = {
            'name': 'Heart_Disease',
            'task': 'binary_classification',
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_classes': 2,
            'description': 'Heart disease diagnosis from clinical measurements'
        }
        
        logger.info(f"Loaded Heart Disease: {metadata['n_samples']} samples, {metadata['n_features']} features")
        return X, y, metadata
        
    except Exception as e:
        logger.warning(f"Failed to load Heart Disease dataset from UCI: {e}")
        logger.info("Using synthetic heart disease-like dataset instead...")
        
        # Generate synthetic heart disease-like data as fallback
        np.random.seed(42)
        n_samples = 303  # Approximate size of real dataset
        
        X = pd.DataFrame({
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'chest_pain': np.random.randint(0, 4, n_samples),
            'resting_bp': np.random.randint(94, 200, n_samples),
            'cholesterol': np.random.randint(126, 564, n_samples),
            'max_hr': np.random.randint(71, 202, n_samples)
        })
        
        # Simple heart disease prediction based on features
        risk_score = (X['age'] * 0.02 + X['chest_pain'] * 0.3 + 
                     (X['resting_bp'] > 140) * 0.2 + (X['cholesterol'] > 240) * 0.2)
        y = pd.Series((risk_score > np.percentile(risk_score, 60)).astype(int), name='heart_disease')
        
        metadata = {
            'name': 'Heart_Disease_Synthetic',
            'task': 'binary_classification',
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_classes': 2,
            'description': 'Synthetic heart disease-like data for diagnosis prediction'
        }
        
        logger.info(f"Generated synthetic Heart Disease: {metadata['n_samples']} samples, {metadata['n_features']} features")
        return X, y, metadata

def get_all_uci_datasets() -> Dict[str, Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]]:
    """
    Load all available UCI datasets.
    
    Returns:
        Dictionary mapping dataset names to (X, y, metadata) tuples
    """
    datasets_dict = {}
    
    dataset_loaders = {
        'covertype': load_covertype,
        'diabetes': load_diabetes_uci,
        'wine_quality': load_wine_quality,
        'adult_census': load_adult_census,
        'heart_disease': load_heart_disease
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
    datasets_dict = get_all_uci_datasets()
    
    print("\n=== UCI DATASETS SUMMARY ===")
    for name, (X, y, metadata) in datasets_dict.items():
        print(f"\n{metadata['name']}:")
        print(f"  Task: {metadata['task']}")
        print(f"  Samples: {metadata['n_samples']}")
        print(f"  Features: {metadata['n_features']}")
        if 'n_classes' in metadata:
            print(f"  Classes: {metadata['n_classes']}")
        print(f"  Description: {metadata['description']}")