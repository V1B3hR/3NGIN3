import numpy as np
import pandas as pd

def generate_linear_data(n_samples=10000, n_features=10, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    coefs = np.random.uniform(-5, 5, size=n_features)
    noise = np.random.normal(0, 0.5, size=n_samples)
    y = X @ coefs + noise
    df = pd.DataFrame(X, columns=[f'num_feature_{i}' for i in range(n_features)])
    df['target_linear'] = y
    return df

def generate_nonlinear_data(n_samples=10000, n_features=10, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = (
        np.sin(X[:, 0]) +
        np.square(X[:, 1]) -
        np.log(np.abs(X[:, 2]) + 1) +
        np.exp(-X[:, 3]) +
        np.random.normal(0, 0.5, size=n_samples)
    )
    df = pd.DataFrame(X, columns=[f'num_feature_{i}' for i in range(n_features)])
    df['target_nonlinear'] = y
    return df

def generate_classification_data(n_samples=10000, n_features=10, n_classes=3, seed=42):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features*0.6),
        n_redundant=int(n_features*0.2),
        n_classes=n_classes,
        random_state=seed
    )
    df = pd.DataFrame(X, columns=[f'num_feature_{i}' for i in range(n_features)])
    df['target_class'] = y
    return df

def generate_mixed_tabular(seed=42):
    # Linear regression data
    linear_df = generate_linear_data(seed=seed)
    # Non-linear regression data
    nonlinear_df = generate_nonlinear_data(seed=seed)
    # Classification data
    class_df = generate_classification_data(seed=seed)
    return linear_df, nonlinear_df, class_df

if __name__ == "__main__":
    linear_df, nonlinear_df, class_df = generate_mixed_tabular()
    print("Linear Regression Data Sample:")
    print(linear_df.head())
    print("\nNon-linear Regression Data Sample:")
    print(nonlinear_df.head())
    print("\nClassification Data Sample:")
    print(class_df.head())
    # Optionally save to CSV if needed:
    # linear_df.to_csv("linear_data.csv", index=False)
    # nonlinear_df.to_csv("nonlinear_data.csv", index=False)
    # class_df.to_csv("classification_data.csv", index=False)
