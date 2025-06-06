import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Utility to load a sample (customize as needed)
def load_sample(sample_idx=0):
    # Load a sample from your preprocessed data (adjust path as needed)
    # For demonstration, we assume a .npy file with shape (N, 800, 1)
    X = np.load('X_val_cnn.npy')
    y = np.load('y_val_cnn.npy')
    return X[sample_idx], y[sample_idx]

def grad_cam_1d(model, sample, class_idx=None, layer_name=None):
    if layer_name is None:
        # Use the last Conv1D layer by default
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                layer_name = layer.name
                break
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(sample, axis=0))
        if class_idx is None:
            class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=0)
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    cam = np.interp(np.arange(sample.shape[0]), np.linspace(0, sample.shape[0], num=cam.shape[0]), cam)
    return cam

def plot_grad_cam(sample, cam, true_label, pred_label, class_names):
    plt.figure(figsize=(12, 4))
    plt.plot(sample.squeeze(), label='Signal')
    plt.imshow(cam[np.newaxis, :], aspect='auto', cmap='jet', alpha=0.5, extent=[0, len(sample), plt.ylim()[0], plt.ylim()[1]])
    plt.title(f'Grad-CAM - True: {class_names[true_label]}, Pred: {class_names[pred_label]}')
    plt.colorbar(label='Importance')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load model and data
    model = load_model('../cnn_model.keras')
    sample, true_label = load_sample(0)
    pred = model.predict(np.expand_dims(sample, axis=0))
    pred_label = np.argmax(pred[0])
    # Class names (update as needed)
    class_names = [
        'Ball_007', 'Ball_014', 'Ball_021',
        'IR_007', 'IR_021',
        'Normal',
        'OR_Centralizado_007', 'OR_Centralizado_014', 'OR_Centralizado_021'
    ]
    cam = grad_cam_1d(model, sample, class_idx=pred_label)
    plot_grad_cam(sample, cam, true_label, pred_label, class_names) 