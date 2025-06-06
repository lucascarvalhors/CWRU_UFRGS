import os
import numpy as np
import pandas as pd
import time
import scipy.stats
import pywt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
                        window_reshaped = window.reshape(-1, 1)  # Formato (200, 1)
                        class_windows[label].append(window_reshaped)
        
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
# DEFINIÇÃO DO MODELO CNN
# ======================================================================

def create_cnn(input_shape, filters=64, kernel_size=3, dense_neurons=64, dropout=0.3, l2_reg=0.001):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
               kernel_regularizer=l2(l2_reg), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout),
        
        Conv1D(filters=filters*2, kernel_size=kernel_size, activation='relu', 
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout),
        
        Flatten(),
        Dense(dense_neurons, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout),
        
        Dense(len(rotulos_cwru), activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def optimize_cnn_fast(filters, kernel_size, dense_neurons, dropout, l2_reg):
    filters = int(filters)
    kernel_size = int(kernel_size)
    dense_neurons = int(dense_neurons)
    
    model = create_cnn((X_train.shape[1], X_train.shape[2]), 
                      filters=filters,
                      kernel_size=kernel_size,
                      dense_neurons=dense_neurons,
                      dropout=dropout,
                      l2_reg=l2_reg)
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    
    history = model.fit(X_sub, y_sub, 
                       epochs=10,
                       batch_size=64,
                       validation_data=(X_val[:len(X_sub)], y_val[:len(X_sub)]),
                       callbacks=[early_stop, reduce_lr],
                       verbose=0)
    
    return max(history.history['val_accuracy'])

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
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    # Verificação dos dados
    print("\nVerificando exemplos de dados por classe:")
    plt.figure(figsize=(15, 5))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.plot(X_train[y_train == i][0])
        plt.title(nomes_classes[i])
    plt.tight_layout()
    plt.show()
    
    # Criar subconjunto para otimização
    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)
    
    # Otimização Bayesiana
    optimizer = BayesianOptimization(
        f=optimize_cnn_fast,
        pbounds={
            'filters': (16, 128),
            'kernel_size': (3, 9),
            'dense_neurons': (32, 256),
            'dropout': (0.2, 0.5),
            'l2_reg': (0.0001, 0.01)
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=10)
    
    # Obter melhores parâmetros
    best_params = optimizer.max['params']
    best_params['filters'] = int(best_params['filters'])
    best_params['kernel_size'] = int(best_params['kernel_size'])
    best_params['dense_neurons'] = int(best_params['dense_neurons'])
    
    print("\nMelhores parâmetros encontrados:")
    print(best_params)
    
    # Treinamento do modelo final
    print("\nIniciando o treinamento do modelo CNN...")
    model = create_cnn((X_train.shape[1], X_train.shape[2]), 
                      filters=best_params['filters'],
                      kernel_size=best_params['kernel_size'],
                      dense_neurons=best_params['dense_neurons'],
                      dropout=best_params['dropout'],
                      l2_reg=best_params['l2_reg'])
    
    # Callbacks atualizados
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, 
                             mode='max', restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', 
                               save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5,
                                min_lr=1e-6, verbose=1)
    
    start_train_time = time.time()
    history = model.fit(X_train, y_train, 
                       epochs=50, 
                       batch_size=128, 
                       validation_data=(X_val, y_val),
                       callbacks=[early_stop, checkpoint, reduce_lr],
                       verbose=1)
    
    train_time = time.time() - start_train_time
    print(f"Tempo total de treinamento: {train_time:.2f} segundos")
    
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
    
    # Avaliação do modelo
    print("\nAvaliando o melhor modelo salvo...")
    best_model = tf.keras.models.load_model('best_model.keras')
    start_val_time = time.time()
    val_loss, val_accuracy = best_model.evaluate(X_val, y_val)
    val_time = time.time() - start_val_time
    print(f"Tempo de avaliação no conjunto de validação: {val_time:.2f} segundos")
    print(f"Perda de validação: {val_loss:.4f}, Acurácia de validação: {val_accuracy:.4f}")

    # Predição e métricas
    y_pred = best_model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nMétricas de desempenho:")
    print(f"Acurácia: {accuracy_score(y_val, y_pred_classes):.4f}")
    print(f"Precisão: {precision_score(y_val, y_pred_classes, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred_classes, average='macro'):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_pred_classes, average='macro'):.4f}")

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
    model.save('cnn_model.keras')

    # Save validation data
    np.save('X_val_cnn.npy', X_val)
    np.save('y_val_cnn.npy', y_val)

except Exception as e:
    print(f"Erro no pipeline: {str(e)}")