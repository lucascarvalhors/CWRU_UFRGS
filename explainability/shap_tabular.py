import numpy as np
import shap
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm

# Feature names for tabular models
feature_names = [
    'mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis', 'rms',
    'peak_to_peak', 'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95',
    'zero_crossing_rate', 'fft_mean', 'fft_std', 'fft_max', 'fft_energy', 'spectral_centroid',
    'wavelet_std_c1', 'wavelet_mean_c1', 'wavelet_std_c2', 'wavelet_mean_c2',
    'wavelet_std_c3', 'wavelet_mean_c3', 'wavelet_std_c4', 'wavelet_mean_c4'
]

# Utility to load validation data (customize as needed)
def load_val_data(model_type='rf'):
    if model_type == 'rf':
        X = np.load('../model_training/X_val_rf.npy')
        y = np.load('../model_training/y_val_rf.npy')
    elif model_type == 'svm':
        X = np.load('../model_training/X_val_svm.npy')
        y = np.load('../model_training/y_val_svm.npy')
    elif model_type == 'mlp':
        X = np.load('../model_training/X_val_mlp.npy')
        y = np.load('../model_training/y_val_mlp.npy')
    else:
        raise ValueError('Unknown model type')
    return X, y

def validate_data(X, y, model_type):
    print(f'Validation for {model_type.upper()}')
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    if X.ndim == 1:
        print('WARNING: X is 1D, only one feature detected!')
    else:
        print(f'Number of features: {X.shape[1]}')
    print('First 5 samples of X:')
    print(X[:5])
    print('First 5 labels of y:')
    print(y[:5])
    if X.ndim > 1 and X.shape[1] == len(feature_names):
        print('Feature names:')
        print(feature_names)
    elif X.ndim > 1:
        print(f'WARNING: Number of features ({X.shape[1]}) does not match expected ({len(feature_names)})!')
    print('-' * 60)

def explain_rf():
    model = joblib.load('../model_training/random_forest_model.joblib')
    X, y = load_val_data('rf')
    validate_data(X, y, 'rf')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"shap_values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"shap_values (list) length: {len(shap_values)}")
        print(f"shap_values[0] shape: {np.array(shap_values[0]).shape}")
    else:
        print(f"shap_values shape: {np.array(shap_values).shape}")
    shap.summary_plot(shap_values, X, feature_names=feature_names if X.shape[1] == len(feature_names) else None, show=True)
    # Also plot global feature importance as a bar plot
    shap.summary_plot(shap_values, X, feature_names=feature_names if X.shape[1] == len(feature_names) else None, plot_type='bar', show=True)

def explain_svm():
    model = joblib.load('../model_training/svm_model.joblib')
    X, y = load_val_data('svm')
    validate_data(X, y, 'svm')
    explainer = shap.KernelExplainer(model.predict_proba, X[:100])
    print('Computing SHAP values for SVM (this may take a while)...')
    shap_values = []
    for i in tqdm(range(len(X[:100]))):
        shap_values.append(explainer.shap_values(X[i:i+1]))
    shap_values = [np.concatenate([sv[class_idx] for sv in shap_values], axis=0) for class_idx in range(len(shap_values[0]))]
    shap.summary_plot(shap_values, X[:100], feature_names=feature_names if X.shape[1] == len(feature_names) else None, show=True)

def explain_mlp():
    model = load_model('../model_training/mlp_model.keras')
    X, y = load_val_data('mlp')
    validate_data(X, y, 'mlp')
    explainer = shap.KernelExplainer(model.predict, X[:100])
    print('Computing SHAP values for MLP (this may take a while)...')
    shap_values = []
    for i in tqdm(range(len(X[:100]))):
        shap_values.append(explainer.shap_values(X[i:i+1]))
    shap_values = [np.concatenate([sv[class_idx] for sv in shap_values], axis=0) for class_idx in range(len(shap_values[0]))]
    shap.summary_plot(shap_values, X[:100], feature_names=feature_names if X.shape[1] == len(feature_names) else None, show=True)

if __name__ == '__main__':
    print('Explaining Random Forest...')
    explain_rf()
    print('Explaining SVM...')
    explain_svm()
    print('Explaining MLP...')
    explain_mlp() 