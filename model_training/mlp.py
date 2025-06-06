import os
import numpy as np
import pandas as pd
import time
import scipy.stats
import pywt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from collections import defaultdict
import random
from bayes_opt import BayesianOptimization
from tensorflow.keras import Input
import zipfile
import gdown

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================================
# CONFIGURAÇÃO DO DATASET CWRU
# ======================================================================

# Dicionário de rótulos para o dataset CWRU
rotulos_cwru = {
    '1797_B_7_DE48.npz': 0,  # Ball_007
    '1797_B_14_DE48.npz': 1,  # Ball_014
    '1797_B_21_DE48.npz': 2,  # Ball_021
    '1797_IR_7_DE48.npz': 3,  # IR_007
    #'1797_IR_14_DE48.npz': 4,  # IR_014
    '1797_IR_21_DE48.npz': 4,  # IR_021
    '1797_Normal.npz': 5,     # Normal
    # Removidas as seguintes classes:
    # '1797_OR@3_7_DE48.npz': 7,  # OR_Ortogonal_007
    # '1797_OR@3_21_DE48.npz': 8,  # OR_Ortogonal_021
    '1797_OR@6_7_DE48.npz': 6,   # OR_Centralizado_007 (reindexado)
    '1797_OR@6_14_DE48.npz': 7, # OR_Centralizado_014 (reindexado)
    '1797_OR@6_21_DE48.npz': 8  # OR_Centralizado_021 (reindexado)
    # Removidas as seguintes classes:
    # '1797_OR@12_7_DE48.npz': 12, # OR_Em_Frente_007
    # '1797_OR@12_21_DE48.npz': 13 # OR_Em_Frente_021
}

# Nomes das classes para exibição (atualizado para refletir as mudanças)
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
# FUNÇÕES DE PRÉ-PROCESSAMENTO (MANTIDAS COM ADAPTAÇÕES)
# ======================================================================

def butter_lowpass_filter(data, cutoff_freq, fs, order=4):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    if normal_cutoff >= 1.0:
        raise ValueError("A frequência de corte normalizada (Wn) deve ser menor que 1.")
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def extract_wavelet_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    features = np.hstack([np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs])
    return features

def calculate_rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def calculate_crest_factor(signal):
    peak = np.max(np.abs(signal))
    rms = calculate_rms(signal)
    return peak / rms if rms != 0 else 0

def calculate_form_factor(signal):
    rms = calculate_rms(signal)
    mean_abs = np.mean(np.abs(signal))
    return rms / mean_abs if mean_abs != 0 else 0

def calculate_fault_factor(signal):
    peak = np.max(np.abs(signal))
    mean_abs = np.mean(np.abs(signal))
    return peak / mean_abs if mean_abs != 0 else 0

def load_cwru_data(sample_size_per_class=None):
    class_windows = defaultdict(list)
    
    # Processar cada arquivo NPZ do CWRU
    for arquivo, label in rotulos_cwru.items():
        caminho = os.path.join(cwru_path, arquivo)
        print(f"Processando arquivo CWRU: {arquivo}")
        
        try:
            data = np.load(caminho)
            arrays_de = [name for name in data.files if 'DE' in name]
            
            for array_name in arrays_de:
                signal = data[array_name].flatten().astype(np.float32)
                
                # Aplicar filtro passa-baixa
                fs = 48000  # Frequência de amostragem do CWRU (48 kHz)
                cutoff_freq = 23900  # Frequência de corte (23.9 kHz)
                try:
                    signal_filtered = butter_lowpass_filter(signal, cutoff_freq, fs)
                except ValueError as e:
                    print(f"Erro ao filtrar o arquivo {arquivo}: {str(e)}")
                    continue
                
                # Janelamento
                window_size = 200
                step_size = window_size * 15 // 100
                
                for i in range(0, len(signal_filtered) - window_size, step_size):
                    window = signal_filtered[i:i+window_size]
                    if len(window) == window_size:
                        # Características estatísticas básicas
                        window_min = np.min(window)
                        window_max = np.max(window)
                        window_mean = np.mean(window)
                        window_std = np.std(window)
                        window_skew = scipy.stats.skew(window)
                        window_kurtosis = scipy.stats.kurtosis(window)
                        
                        # Novas características adicionadas
                        window_rms = calculate_rms(window)
                        window_crest_factor = calculate_crest_factor(window)
                        window_form_factor = calculate_form_factor(window)
                        window_fault_factor = calculate_fault_factor(window)
                        
                        # FFT e características da wavelet
                        fft_result = np.abs(fft(window)).astype(np.float32)
                        wavelet_features = extract_wavelet_features(window)
                        mean_fft = np.mean(fft_result)
                        median_fft = np.median(fft_result)
                        
                        # Combinando todas as características
                        combined_features = np.concatenate((
                            window,
                            wavelet_features,
                            [
                                mean_fft, median_fft, 
                                window_min, window_max, window_mean, window_std, 
                                window_skew, window_kurtosis,
                                window_rms, window_crest_factor,
                                window_form_factor, window_fault_factor
                            ]
                        )).astype(np.float32)
                        
                        class_windows[label].append(combined_features)
        
        except Exception as e:
            print(f"Erro ao processar arquivo {arquivo}: {str(e)}")
            continue
    
    # Balanceamento das classes (se especificado)
    X, y = [], []
    for class_label, windows in class_windows.items():
        sampled_windows = random.sample(windows, min(len(windows), sample_size_per_class)) if sample_size_per_class else windows
        X.extend(sampled_windows)
        y.extend([class_label] * len(sampled_windows))
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)  # y como int32

# ======================================================================
# DEFINIÇÃO DO MODELO (MANTIDA)
# ======================================================================

def create_mlp(input_dim, neurons=64, dropout=0.3, l2_reg=0.001):
    model = Sequential([
        Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(len(rotulos_cwru), activation='softmax')  # Ajustado para as classes do CWRU
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def optimize_mlp(neurons, dropout, l2_reg):
    model = create_mlp(X_train.shape[1], int(neurons), dropout, l2_reg)
    history = model.fit(X_train, y_train, epochs=10, batch_size=1024, validation_data=(X_val, y_val), verbose=0)
    return -min(history.history['val_loss'])

# ======================================================================
# PIPELINE PRINCIPAL
# ======================================================================

try:
    # Carregar e pré-processar dados CWRU
    X, y = load_cwru_data(sample_size_per_class=5000)  # Você pode ajustar este número
    
    # Adicionado: Mostrar número de amostras por classe
    print("\nNúmero de amostras por classe:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"{nomes_classes[class_id]}: {count} amostras")
    
    # Dividir os dados
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1024, stratify=y)
    
    # Adicionado: Mostrar distribuição após split
    print("\nNúmero de amostras no treino/validação por classe:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    for class_id, count_train, count_val in zip(unique_train, counts_train, counts_val):
        print(f"{nomes_classes[class_id]}: Treino={count_train} | Val={count_val}")
    
    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Otimização Bayesiana
    optimizer = BayesianOptimization(
        f=optimize_mlp,
        pbounds={
            'neurons': (32, 1024),
            'dropout': (0.1, 0.5),
            'l2_reg': (0.0001, 0.01)
        },
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=10)
    best_params = optimizer.max['params']
    
    # Treinamento do modelo
    print("Iniciando o treinamento do modelo...")
    start_train_time = time.time()
    model = create_mlp(X_train.shape[1], int(best_params['neurons']), best_params['dropout'], best_params['l2_reg'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=1024, validation_data=(X_val, y_val), verbose=1)
    train_time = time.time() - start_train_time
    print(f"Tempo total de treinamento: {train_time:.2f} segundos")
    
    # Plotar histórico de treinamento
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Evolução da Perda Durante o Treinamento')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Evolução da Acurácia Durante o Treinamento')
    plt.legend()
    plt.show()
    
    # Avaliação do modelo
    start_val_time = time.time()
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    val_time = time.time() - start_val_time
    print(f"Tempo de avaliação no conjunto de validação: {val_time:.2f} segundos")
    print(f"Perda de validação: {val_loss:.4f}, Acurácia de validação: {val_accuracy:.4f}")

    # Predição
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Métricas
    print("\nMétricas de desempenho:")
    print(f"Acurácia: {accuracy_score(y_val, y_pred_classes):.4f}")
    print(f"Precisão: {precision_score(y_val, y_pred_classes, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred_classes, average='macro'):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_pred_classes, average='macro'):.4f}")

    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_val, y_pred_classes, target_names=nomes_classes))

    # Matriz de confusão
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()

    # After training the final model
    model.save('mlp_model.keras')

    # Save validation data
    np.save('X_val_mlp.npy', X_val)
    np.save('y_val_mlp.npy', y_val)

except Exception as e:
    print(f"Erro no pipeline: {str(e)}")