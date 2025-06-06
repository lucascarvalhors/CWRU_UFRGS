import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Utility to load a sample (customize as needed)
def load_sample(sample_idx=0):
    X = np.load('../model_training/X_val_cnn.npy')
    y = np.load('../model_training/y_val_cnn.npy')
    return X[sample_idx], y[sample_idx]

def compute_saliency(model, sample, class_idx=None):
    sample = tf.convert_to_tensor(np.expand_dims(sample, axis=0))
    with tf.GradientTape() as tape:
        tape.watch(sample)
        preds = model(sample)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, sample)[0].numpy().squeeze()
    saliency = np.abs(grads)
    return saliency

def plot_saliency(sample, saliency, true_label, pred_label, class_names, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(sample.squeeze(), label='Signal')
    plt.plot(saliency, label='Saliency', alpha=0.7)
    plt.title(f'Saliency Map - True: {class_names[true_label]}, Pred: {class_names[pred_label]}')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    model = load_model('../model_training/cnn_model.keras')
    X = np.load('../model_training/X_val_cnn.npy')
    y = np.load('../model_training/y_val_cnn.npy')
    class_names = [
        'Ball_007', 'Ball_014', 'Ball_021',
        'IR_007', 'IR_021',
        'Normal',
        'OR_Centralizado_007', 'OR_Centralizado_014', 'OR_Centralizado_021'
    ]
    results_dir = 'saliency_results'
    os.makedirs(results_dir, exist_ok=True)
    for class_idx in range(len(class_names)):
        sample_indices = np.where(y == class_idx)[0]
        if len(sample_indices) == 0:
            print(f'No sample found for class {class_names[class_idx]}')
            continue
        sample_idx = sample_indices[0]
        sample = X[sample_idx]
        true_label = y[sample_idx]
        pred = model.predict(np.expand_dims(sample, axis=0))
        pred_label = np.argmax(pred[0])
        saliency = compute_saliency(model, sample, class_idx=pred_label)
        save_path = os.path.join(results_dir, f'saliency_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png')
        plot_saliency(sample, saliency, true_label, pred_label, class_names, save_path=save_path)
        print(f'Saved saliency map for class {class_names[class_idx]} to {save_path}') 