import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Utility to load validation data (customize as needed)
def load_val_data(model_type='rf'):
    if model_type == 'rf':
        X = np.load('../model_training/X_val_rf.npy')
        y = np.load('../model_training/y_val_rf.npy')
    elif model_type == 'svm':
        X = np.load('../model_training/X_val_svm.npy')
        y = np.load('../model_training/y_val_svm.npy')
    else:
        raise ValueError('Unknown model type')
    return X, y

def plot_importance(importances, feature_names, title):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.barh(np.array(feature_names)[indices], importances[indices])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def explain_rf():
    model = joblib.load('../model_training/random_forest_model.joblib')
    X, y = load_val_data('rf')
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    feature_names = [
        'mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis', 'rms',
        'peak_to_peak', 'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95',
        'zero_crossing_rate', 'fft_mean', 'fft_std', 'fft_max', 'fft_energy', 'spectral_centroid',
        'wavelet_std_c1', 'wavelet_mean_c1', 'wavelet_std_c2', 'wavelet_mean_c2',
        'wavelet_std_c3', 'wavelet_mean_c3', 'wavelet_std_c4', 'wavelet_mean_c4'
    ]
    plot_importance(result.importances_mean, feature_names, 'Permutation Importance - Random Forest')

def explain_svm():
    model = joblib.load('../model_training/svm_model.joblib')
    X, y = load_val_data('svm')
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    feature_names = [
        'mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis', 'rms',
        'peak_to_peak', 'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95',
        'zero_crossing_rate', 'fft_mean', 'fft_std', 'fft_max', 'fft_energy', 'spectral_centroid',
        'wavelet_std_c1', 'wavelet_mean_c1', 'wavelet_std_c2', 'wavelet_mean_c2',
        'wavelet_std_c3', 'wavelet_mean_c3', 'wavelet_std_c4', 'wavelet_mean_c4'
    ]
    plot_importance(result.importances_mean, feature_names, 'Permutation Importance - SVM')

if __name__ == '__main__':
    print('Permutation Importance for Random Forest...')
    explain_rf()
    print('Permutation Importance for SVM...')
    explain_svm() 