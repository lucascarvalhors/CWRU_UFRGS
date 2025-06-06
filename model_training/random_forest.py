import os
import numpy as np
import pandas as pd
import time
import scipy.stats
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from collections import defaultdict
import random
import warnings
import zipfile
import gdown
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================================
# CONFIGURAÇÃO DO DATASET CWRU
# ======================================================================

rotulos_cwru = {
    '1797_B_7_DE48.npz': 0,  # Ball_007
    '1797_B_14_DE48.npz': 1,  # Ball_014
    '1797_B_21_DE48.npz': 2,  # Ball_021
    '1797_IR_7_DE48.npz': 3,  # IR_007
    '1797_IR_21_DE48.npz': 4, # IR_021
    '1797_Normal.npz': 5,     # Normal
    '1797_OR@6_7_DE48.npz': 6,   # OR_Centralizado_007
    '1797_OR@6_14_DE48.npz': 7,  # OR_Centralizado_014
    '1797_OR@6_21_DE48.npz': 8   # OR_Centralizado_021
}

nomes_classes = [
    'Ball_007', 'Ball_014', 'Ball_021',
    'IR_007', 'IR_021',
    'Normal',
    'OR_Centralizado_007', 'OR_Centralizado_014', 'OR_Centralizado_021'
]

# Path to store the dataset
local_data_dir = os.path.join(os.path.expanduser('~'), 'CWRU_Bearing_Data')
local_zip_path = os.path.join(local_data_dir, 'CWRU_Bearing_NumPy.zip')
extracted_data_dir = os.path.join(local_data_dir, 'CWRU_Bearing_NumPy-main', 'Data', '1797 RPM')

def download_and_extract_cwru():
    if not os.path.exists(extracted_data_dir):
        os.makedirs(local_data_dir, exist_ok=True)
        if not os.path.exists(local_zip_path):
            print('Downloading CWRU dataset...')
            gdown.download('https://drive.google.com/uc?id=1l-P6Nlzh_5-JKy8GzKIBgYJYUy4qZbKA', local_zip_path, quiet=False)
        print('Extracting CWRU dataset...')
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_data_dir)
        print('Extraction complete.')
    else:
        print('CWRU dataset already available.')

download_and_extract_cwru()

cwru_path = extracted_data_dir

# ======================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# ======================================================================

def butter_lowpass_filter(data, cutoff_freq, fs, order=4):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    if normal_cutoff >= 1.0:
        raise ValueError("A frequência de corte normalizada (Wn) deve ser menor que 1.")
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def extract_features(signal):
    """Extract features from time series signal for Random Forest"""
    features = []
    
    # Time-domain features
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(np.median(signal))
    features.append(scipy.stats.skew(signal))
    features.append(scipy.stats.kurtosis(signal))
    features.append(np.sqrt(np.mean(signal**2)))  # RMS
    
    # Peak-to-peak amplitude
    features.append(np.max(signal) - np.min(signal))
    
    # Percentiles
    features.append(np.percentile(signal, 5))
    features.append(np.percentile(signal, 25))
    features.append(np.percentile(signal, 75))
    features.append(np.percentile(signal, 95))
    
    # Zero crossing rate
    features.append(np.sum((signal[:-1] * signal[1:]) < 0))
    
    # Frequency-domain features (using FFT)
    fft_vals = np.abs(fft(signal))
    fft_freq = np.fft.fftfreq(len(signal))
    positive_freq = fft_freq > 0
    fft_vals = fft_vals[positive_freq]
    
    features.append(np.mean(fft_vals))
    features.append(np.std(fft_vals))
    features.append(np.max(fft_vals))
    features.append(np.sum(fft_vals**2))  # Energy
    
    # Spectral centroid
    if np.sum(fft_vals) > 0:
        features.append(np.sum(fft_freq[positive_freq] * fft_vals) / np.sum(fft_vals))
    else:
        features.append(0.0)
    
    # Wavelet features
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    for i in range(1, len(coeffs)):
        features.append(np.std(coeffs[i]))
        features.append(np.mean(np.abs(coeffs[i])))
    
    return np.array(features)

def load_cwru_data(sample_size_per_class=None):
    class_windows = defaultdict(list)
    
    for arquivo, label in rotulos_cwru.items():
        caminho = os.path.join(cwru_path, arquivo)
        print(f"Processando arquivo CWRU: {arquivo}")
        
        try:
            data = np.load(caminho)
            arrays_de = [name for name in data.files if 'DE' in name]
            
            for array_name in arrays_de:
                signal = data[array_name].flatten().astype(np.float32)
                
                fs = 48000
                cutoff_freq = 23900
                try:
                    signal_filtered = butter_lowpass_filter(signal, cutoff_freq, fs)
                except ValueError as e:
                    print(f"Erro ao filtrar o arquivo {arquivo}: {str(e)}")
                    continue
                
                window_size = 800
                step_size = window_size * 15 // 100
                
                for i in range(0, len(signal_filtered) - window_size, step_size):
                    window = signal_filtered[i:i+window_size]
                    if len(window) == window_size:
                        features = extract_features(window)
                        class_windows[label].append(features)
        
        except Exception as e:
            print(f"Erro ao processar arquivo {arquivo}: {str(e)}")
            continue
    
    X, y = [], []
    for class_label, windows in class_windows.items():
        sampled_windows = random.sample(windows, min(len(windows), sample_size_per_class)) if sample_size_per_class else windows
        X.extend(sampled_windows)
        y.extend([class_label] * len(sampled_windows))
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# ======================================================================
# PIPELINE PRINCIPAL
# ======================================================================

try:
    # Carregar e pré-processar dados CWRU
    X, y = load_cwru_data(sample_size_per_class=5000)
    
    print("\nNúmero de amostras por classe:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"{nomes_classes[class_id]}: {count} amostras")
    
    # Dividir os dados
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1024, stratify=y)
    
    print("\nNúmero de amostras no treino/validação por classe:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    for class_id, count_train, count_val in zip(unique_train, counts_train, counts_val):
        print(f"{nomes_classes[class_id]}: Treino={count_train} | Val={count_val}")
    
    # Normalização global
    print("\nAplicando normalização global...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Treinamento do modelo Random Forest
    print("\nIniciando o treinamento do modelo Random Forest...")
    
    # Parâmetros para busca em grade
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Criar modelo Random Forest
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Busca em grade com validação cruzada
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
    
    start_train_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_train_time
    
    print(f"Tempo total de treinamento: {train_time:.2f} segundos")
    print("\nMelhores parâmetros encontrados:")
    print(grid_search.best_params_)
    
    # Obter o melhor modelo
    best_rf = grid_search.best_estimator_
    
    # Configurar estilo padrão do Matplotlib para evitar fundos cinza
    plt.style.use('default')
    
    # Avaliação do modelo
    print("\nAvaliando o melhor modelo...")
    start_val_time = time.time()
    val_accuracy = best_rf.score(X_val, y_val)
    val_time = time.time() - start_val_time
    print(f"Tempo de avaliação no conjunto de validação: {val_time:.2f} segundos")
    print(f"Acurácia de validação: {val_accuracy:.4f}")

    # Predição e métricas
    y_pred = best_rf.predict(X_val)

    print("\nMétricas de desempenho:")
    print(f"Acurácia: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_val, y_pred, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred, average='macro'):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_pred, average='macro'):.4f}")

    print("\nRelatório de Classificação:")
    print(classification_report(y_val, y_pred, target_names=nomes_classes))

    # Matriz de confusão
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão - Random Forest')
    plt.show()
    
    # Feature importance plot - ordenado
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    feature_names = [
        'mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis', 'rms',
        'peak_to_peak', 'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95',
        'zero_crossing_rate', 'fft_mean', 'fft_std', 'fft_max', 'fft_energy', 'spectral_centroid',
        'wavelet_std_c1', 'wavelet_mean_c1', 'wavelet_std_c2', 'wavelet_mean_c2',
        'wavelet_std_c3', 'wavelet_mean_c3', 'wavelet_std_c4', 'wavelet_mean_c4'
    ]

    plt.figure(figsize=(12, 6))
    plt.title("Importância das Features")
    plt.barh(feature_names, importances[indices])
    plt.xlabel("Importância")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

    # After training the best_rf model
    joblib.dump(best_rf, 'random_forest_model.joblib')

    # Save validation data
    np.save('X_val_rf.npy', X_val)
    np.save('y_val_rf.npy', y_val)

except Exception as e:
    print(f"Erro no pipeline: {str(e)}")
