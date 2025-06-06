import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Utility to convert Sequential model to Functional
def sequential_to_functional(sequential_model):
    if isinstance(sequential_model, tf.keras.Model) and not isinstance(sequential_model, tf.keras.Sequential):
        print('Model is already functional.')
        return sequential_model
    input_layer = tf.keras.Input(shape=sequential_model.input_shape[1:])
    x = input_layer
    for layer in sequential_model.layers:
        x = layer(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)

# Utility to load a sample (customize as needed)
def load_sample(sample_idx=0):
    # Load a sample from your preprocessed data (adjust path as needed)
    # For demonstration, we assume a .npy file with shape (N, 800, 1)
    X = np.load('../model_training/X_val_cnn.npy')
    y = np.load('../model_training/y_val_cnn.npy')
    return X[sample_idx], y[sample_idx]

def grad_cam_1d(model, sample, class_idx=None, layer_name=None):
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                layer_name = layer.name
                break
    input_data = tf.convert_to_tensor(np.expand_dims(sample, axis=0), dtype=tf.float32)
    # Force model build
    _ = model(input_data)
    with tf.GradientTape() as tape:
        intermediate_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        conv_outputs, predictions = intermediate_model(input_data)
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

def plot_grad_cam(sample, cam, true_label, pred_label, class_names, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(sample.squeeze(), label='Signal')
    plt.imshow(cam[np.newaxis, :], aspect='auto', cmap='jet', alpha=0.5, extent=[0, len(sample), plt.ylim()[0], plt.ylim()[1]])
    plt.title(f'Grad-CAM - True: {class_names[true_label]}, Pred: {class_names[pred_label]}')
    plt.colorbar(label='Importance')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Load model and data
    model = load_model('../model_training/cnn_model.keras')
    model = sequential_to_functional(model)
    X = np.load('../model_training/X_val_cnn.npy')
    y = np.load('../model_training/y_val_cnn.npy')
    # Class names (update as needed)
    class_names = [
        'Ball_007', 'Ball_014', 'Ball_021',
        'IR_007', 'IR_021',
        'Normal',
        'OR_Centralizado_007', 'OR_Centralizado_014', 'OR_Centralizado_021'
    ]
    results_dir = 'grad_cam_results'
    os.makedirs(results_dir, exist_ok=True)
    for class_idx in range(len(class_names)):
        # Find the first sample of this class
        sample_indices = np.where(y == class_idx)[0]
        if len(sample_indices) == 0:
            print(f'No sample found for class {class_names[class_idx]}')
            continue
        sample_idx = sample_indices[0]
        sample = X[sample_idx]
        true_label = y[sample_idx]
        pred = model.predict(np.expand_dims(sample, axis=0))
        pred_label = np.argmax(pred[0])
        cam = grad_cam_1d(model, sample, class_idx=pred_label)
        save_path = os.path.join(results_dir, f'grad_cam_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png')
        plot_grad_cam(sample, cam, true_label, pred_label, class_names, save_path=save_path)
        print(f'Saved Grad-CAM for class {class_names[class_idx]} to {save_path}') 