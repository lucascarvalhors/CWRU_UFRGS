import os
import numpy as np
import pandas as pd
import time
import scipy.stats
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from collections import defaultdict
import random
import warnings

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

cwru_path = r'C:\Users\LCARVA21\Pictures\CWRU_Bearing_NumPy-main\Data\1797 RPM'

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
    """Extract features from time series signal for SVM"""
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
    
    # Frequency-domain features (using FFT)
    fft_vals = np.abs(fft(signal))
    fft_freq = np.fft.fftfreq(len(signal))
    positive_freq = fft_freq > 0
    fft_vals = fft_vals[positive_freq]
    
    features.append(np.mean(fft_vals))
    features.append(np.std(fft_vals))
    features.append(np.max(fft_vals))
    features.append(np.sum(fft_vals**2))  # Energy
    
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
    
    # Treinamento do modelo SVM
    print("\nIniciando o treinamento do modelo SVM...")
    
    # Parâmetros para busca em grade
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Criar modelo SVM
    svm = SVC(decision_function_shape='ovr', probability=True)
    
    # Busca em grade com validação cruzada
    grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=2)
    
    start_train_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_train_time
    
    print(f"Tempo total de treinamento: {train_time:.2f} segundos")
    print("\nMelhores parâmetros encontrados:")
    print(grid_search.best_params_)
    
    # Configurar estilo padrão do Matplotlib para evitar fundos cinza
    plt.style.use('default')

    # Plotar histórico de treinamento
    fig = plt.figure(figsize=(12, 5), facecolor='white')  # Define fundo da figura como branco

    # Gráfico de Perda
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('white')  # Define fundo do eixo como branco
    ax1.set_axisbelow(True)  # Garante que a grade fique atrás das linhas
    plt.plot(history.history['loss'], label='Perda de Treinamento', linewidth=2)
    plt.plot(history.history['val_loss'], label='Perda de Validação', linewidth=2)
    plt.title('Evolução da Perda Durante o Treinamento', fontsize=12)
    plt.xlabel('Época', fontsize=10)
    plt.ylabel('Perda', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7, color='lightgray')  # Grade clara para contraste
    plt.legend(fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Gráfico de Acurácia
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('white')  # Define fundo do eixo como branco
    ax2.set_axisbelow(True)  # Garante que a grade fique atrás das linhas
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação', linewidth=2)
    plt.title('Evolução da Acurácia Durante o Treinamento', fontsize=12)
    plt.xlabel('Época', fontsize=10)
    plt.ylabel('Acurácia', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7, color='lightgray')  # Grade clara para contraste
    plt.legend(fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.show()
    
    # Obter o melhor modelo
    best_svm = grid_search.best_estimator_
    
    # Avaliação do modelo
    print("\nAvaliando o melhor modelo...")
    start_val_time = time.time()
    val_accuracy = best_svm.score(X_val, y_val)
    val_time = time.time() - start_val_time
    print(f"Tempo de avaliação no conjunto de validação: {val_time:.2f} segundos")
    print(f"Acurácia de validação: {val_accuracy:.4f}")

    # Predição e métricas
    y_pred = best_svm.predict(X_val)

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
    plt.title('Matriz de Confusão - SVM')
    plt.show()

except Exception as e:
    print(f"Erro no pipeline: {str(e)}")