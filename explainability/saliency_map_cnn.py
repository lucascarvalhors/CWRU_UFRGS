import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Utility to load a sample (customize as needed)
def load_sample(sample_idx=0):
    X = np.load('X_val_cnn.npy')
    y = np.load('y_val_cnn.npy')
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

def plot_saliency(sample, saliency, true_label, pred_label, class_names):
    plt.figure(figsize=(12, 4))
    plt.plot(sample.squeeze(), label='Signal')
    plt.plot(saliency, label='Saliency', alpha=0.7)
    plt.title(f'Saliency Map - True: {class_names[true_label]}, Pred: {class_names[pred_label]}')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = load_model('../cnn_model.keras')
    sample, true_label = load_sample(0)
    pred = model.predict(np.expand_dims(sample, axis=0))
    pred_label = np.argmax(pred[0])
    class_names = [
        'Ball_007', 'Ball_014', 'Ball_021',
        'IR_007', 'IR_021',
        'Normal',
        'OR_Centralizado_007', 'OR_Centralizado_014', 'OR_Centralizado_021'
    ]
    saliency = compute_saliency(model, sample, class_idx=pred_label)
    plot_saliency(sample, saliency, true_label, pred_label, class_names) 