"""
Kaggle Dataset Loaders for 3NGIN3 Training

This module provides access to popular Kaggle competition datasets for testing
the 3NGIN3 reasoning modes and evaluation capabilities.

Supported datasets:
- Titanic - Binary classification (survival prediction)
- House Prices - Regression (price prediction)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import requests
import io
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_titanic() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the Titanic dataset for survival prediction.
    
    Returns:
        X: Features DataFrame
        y: Target Series (1 = Survived, 0 = Did not survive)
        metadata: Dataset information
    """
    logger.info("Loading Titanic dataset...")
    
    try:
        # Try to get Titanic data from a public source
        # Using seaborn's built-in Titanic dataset if available
        import seaborn as sns
        data = sns.load_dataset('titanic')
        
        # Clean and prepare the data
        data = data.dropna(subset=['age', 'embarked'])
        
        # Feature engineering
        data['family_size'] = data['sibsp'] + data['parch'] + 1
        data['is_alone'] = (data['family_size'] == 1).astype(int)
        data['title'] = data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Simplify titles
        data['title'] = data['title'].replace(['Lady', 'Countess','Capt', 'Col',
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data['title'] = data['title'].replace('Mlle', 'Miss')
        data['title'] = data['title'].replace('Ms', 'Miss')
        data['title'] = data['title'].replace('Mme', 'Mrs')
        
        # Encode categorical variables
        categorical_cols = ['sex', 'embarked', 'title']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        # Select features
        feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 
                       'embarked', 'family_size', 'is_alone', 'title']
        
        X = data[feature_cols]
        y = data['survived']
        y.name = 'survived'
        
        metadata = {
            'name': 'Titanic',
            'task': 'binary_classification',
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_classes': 2,
            'description': 'Predict survival on the Titanic based on passenger characteristics'
        }
        
        logger.info(f"Loaded Titanic: {metadata['n_samples']} samples, {metadata['n_features']} features")
        return X, y, metadata
        
    except ImportError:
        logger.warning("Seaborn not available, generating synthetic Titanic-like dataset...")
        
    except Exception as e:
        logger.warning(f"Failed to load real Titanic dataset: {e}")
        logger.info("Generating synthetic Titanic-like dataset...")
    
    # Generate synthetic Titanic-like data as fallback
    np.random.seed(42)
    n_samples = 891  # Size of real Titanic training set
    
    # Generate realistic Titanic features
    pclass = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
    sex = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])  # 0=female, 1=male
    age = np.random.normal(29, 14, n_samples)
    age = np.clip(age, 0.42, 80)  # Clip to realistic age range
    
    sibsp = np.random.choice([0, 1, 2, 3, 4, 5, 8], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.003, 0.002])
    parch = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.02, 0.004, 0.003, 0.003])
    
    # Fare depends on class
    fare = np.zeros(n_samples)
    fare[pclass == 1] = np.random.gamma(2, 20, sum(pclass == 1))  # Higher fares for 1st class
    fare[pclass == 2] = np.random.gamma(2, 10, sum(pclass == 2))  # Medium fares for 2nd class  
    fare[pclass == 3] = np.random.gamma(1, 8, sum(pclass == 3))   # Lower fares for 3rd class
    
    embarked = np.random.choice([0, 1, 2], n_samples, p=[0.65, 0.19, 0.16])  # S, C, Q
    
    X = pd.DataFrame({
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked,
        'family_size': sibsp + parch + 1,
        'is_alone': (sibsp + parch == 0).astype(int)
    })
    
    # Survival based on historical patterns
    survival_prob = (
        (X['sex'] == 0) * 0.7 +  # Women more likely to survive
        (X['pclass'] == 1) * 0.4 +  # First class more likely to survive
        (X['pclass'] == 2) * 0.2 +  # Second class moderate survival
        (X['age'] < 15) * 0.3 +  # Children more likely to survive
        np.random.normal(0, 0.1, n_samples)  # Random variation
    )
    
    y = pd.Series((survival_prob > 0.5).astype(int), name='survived')
    
    metadata = {
        'name': 'Titanic_Synthetic',
        'task': 'binary_classification',
        'n_samples': len(X),
        'n_features': len(X.columns),
        'n_classes': 2,
        'description': 'Synthetic Titanic-like data for survival prediction'
    }
    
    logger.info(f"Generated synthetic Titanic: {metadata['n_samples']} samples, {metadata['n_features']} features")
    return X, y, metadata

def load_house_prices() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the House Prices dataset for price prediction.
    
    Returns:
        X: Features DataFrame
        y: Target Series (house prices)
        metadata: Dataset information
    """
    logger.info("Loading House Prices dataset...")
    
    # Generate synthetic house prices data (since real Kaggle data requires download)
    logger.info("Generating synthetic House Prices dataset...")
    
    np.random.seed(42)
    n_samples = 1460  # Size of real House Prices training set
    
    # Generate realistic house features
    lot_frontage = np.random.normal(70, 20, n_samples)
    lot_frontage = np.clip(lot_frontage, 21, 313)
    
    lot_area = np.random.normal(10000, 5000, n_samples)
    lot_area = np.clip(lot_area, 1300, 215245)
    
    year_built = np.random.randint(1872, 2010, n_samples)
    year_remod = np.maximum(year_built, np.random.randint(1950, 2010, n_samples))
    
    total_bsmt_sf = np.random.normal(1000, 400, n_samples)
    total_bsmt_sf = np.clip(total_bsmt_sf, 0, 6110)
    
    first_flr_sf = np.random.normal(1200, 400, n_samples)
    first_flr_sf = np.clip(first_flr_sf, 334, 4692)
    
    gr_liv_area = first_flr_sf + np.random.normal(300, 200, n_samples)
    gr_liv_area = np.clip(gr_liv_area, 334, 5642)
    
    full_bath = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.02, 0.15, 0.65, 0.16, 0.02])
    bedroom_probs = [0.01, 0.03, 0.13, 0.55, 0.25, 0.025, 0.008, 0.002]
    bedroom_probs = np.array(bedroom_probs) / sum(bedroom_probs)  # Normalize
    bedroom_abvgr = np.random.choice([0, 1, 2, 3, 4, 5, 6, 8], n_samples, p=bedroom_probs)
    
    garage_cars = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.06, 0.18, 0.59, 0.15, 0.015, 0.005])
    garage_area = garage_cars * np.random.normal(300, 100, n_samples)
    garage_area = np.clip(garage_area, 0, 1418)
    
    overall_qual = np.random.choice(range(1, 11), n_samples, p=[0.02, 0.03, 0.04, 0.08, 0.20, 0.22, 0.20, 0.12, 0.07, 0.02])
    probs = [0.005, 0.01, 0.02, 0.06, 0.57, 0.25, 0.08, 0.02, 0.003, 0.002]
    probs = np.array(probs) / sum(probs)  # Normalize to sum to 1
    overall_cond = np.random.choice(range(1, 11), n_samples, p=probs)
    
    X = pd.DataFrame({
        'lot_frontage': lot_frontage,
        'lot_area': lot_area,
        'year_built': year_built,
        'year_remod_add': year_remod,
        'total_bsmt_sf': total_bsmt_sf,
        'first_flr_sf': first_flr_sf,
        'gr_liv_area': gr_liv_area,
        'full_bath': full_bath,
        'bedroom_abvgr': bedroom_abvgr,
        'garage_cars': garage_cars,
        'garage_area': garage_area,
        'overall_qual': overall_qual,
        'overall_cond': overall_cond
    })
    
    # Price based on realistic factors
    base_price = (
        X['gr_liv_area'] * 100 +
        X['overall_qual'] * 15000 +
        X['overall_cond'] * 5000 +
        X['garage_area'] * 50 +
        X['total_bsmt_sf'] * 30 +
        (2010 - X['year_built']) * (-200) +  # Newer homes more expensive
        X['full_bath'] * 5000 +
        X['garage_cars'] * 8000 +
        np.random.normal(0, 20000, n_samples)  # Random variation
    )
    
    # Ensure positive prices
    y = pd.Series(np.maximum(base_price, 34900), name='sale_price')
    
    metadata = {
        'name': 'House_Prices_Synthetic',
        'task': 'regression',
        'n_samples': len(X),
        'n_features': len(X.columns),
        'description': 'Synthetic house prices data for price prediction'
    }
    
    logger.info(f"Generated synthetic House Prices: {metadata['n_samples']} samples, {metadata['n_features']} features")
    return X, y, metadata

def get_all_kaggle_datasets() -> Dict[str, Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]]:
    """
    Load all available Kaggle datasets.
    
    Returns:
        Dictionary mapping dataset names to (X, y, metadata) tuples
    """
    datasets_dict = {}
    
    dataset_loaders = {
        'titanic': load_titanic,
        'house_prices': load_house_prices
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
    datasets_dict = get_all_kaggle_datasets()
    
    print("\n=== KAGGLE DATASETS SUMMARY ===")
    for name, (X, y, metadata) in datasets_dict.items():
        print(f"\n{metadata['name']}:")
        print(f"  Task: {metadata['task']}")
        print(f"  Samples: {metadata['n_samples']}")
        print(f"  Features: {metadata['n_features']}")
        if 'n_classes' in metadata:
            print(f"  Classes: {metadata['n_classes']}")
        print(f"  Description: {metadata['description']}")
        print(f"  Target range: {y.min():.2f} - {y.max():.2f}")