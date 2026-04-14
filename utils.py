import numpy as np

class StandardScaler:
    """
    Manual StandardScaler — no sklearn dependency.
    Used in both Lambda training and Streamlit dashboard.
    """
    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X,  axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_