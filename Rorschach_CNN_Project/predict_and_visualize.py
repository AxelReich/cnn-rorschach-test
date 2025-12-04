import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib 
import cv2
import os
import data_pipeline as data

# Constants
IMG_SIZE = (224, 224)
MODEL_DIR = 'saved_models'
DATA_DIR = 'data/rorschach_blots'
OUTPUT_DIR = 'experiment_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_img_array(img_path, size):
    # Load image
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = array / 255.0
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if len(preds.shape) == 4:
            preds = tf.reduce_mean(preds, axis=(1, 2))
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generate heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps['jet']
    
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    print(f"Saved visualization to: {cam_path}")

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None

def main():
    print("=== Loading Models... ===")
    try:
        model_a = keras.models.load_model(os.path.join(MODEL_DIR, 'model_A_scratch.keras'))
        model_b = keras.models.load_model(os.path.join(MODEL_DIR, 'model_B_transfer.keras'))
        print("Models loaded successfully.")
    except OSError:
        print("ERROR: Could not load models. Did you run 'train.py'?")
        return

    _, _, _ = data.get_data_generators()

    layer_name_a = find_last_conv_layer(model_a)
    print(f"Target Layer for Model A: {layer_name_a}")

    layer_name_b = 'out_relu'
    print(f"Target Layer for Model B: {layer_name_b}")
    rorschach_images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))])


    print(f"\nProcessing {len(rorschach_images)} images")

    for img_name in rorschach_images:
        img_path = os.path.join(DATA_DIR, img_name)
        img_array = get_img_array(img_path, IMG_SIZE)
        
        print(f"\n--- Analyzing {img_name} ---")

        # Model A heatmap
        preds_a = model_a.predict(img_array)
        idx_a = np.argmax(preds_a[0])
        print(f"{idx_a} ({100*np.max(preds_a):.2f}%)")
        
        # Model A visualization
        heatmap_a = make_gradcam_heatmap(img_array, model_a, layer_name_a)
        save_name_a = os.path.join(OUTPUT_DIR, f"{img_name}_ModelA.jpg")
        save_and_display_gradcam(img_path, heatmap_a, save_name_a)

        # Model B visualization
        mobile_net_base = None
        for layer in model_b.layers:
            # Search the mobile net 
            if 'mobilenet' in layer.name.lower():
                mobile_net_base = layer
                break
        
        if mobile_net_base:
            heatmap_b = make_gradcam_heatmap(img_array, mobile_net_base, layer_name_b)
        else:
            heatmap_b = make_gradcam_heatmap(img_array, model_b, layer_name_b)
            
        preds_b = model_b.predict(img_array)
        idx_b = np.argmax(preds_b[0])
        print(f"Model B sees: Class {idx_b} ({100*np.max(preds_b):.2f}%)")
        
        save_name_b = os.path.join(OUTPUT_DIR, f"{img_name}_ModelB.jpg")
        save_and_display_gradcam(img_path, heatmap_b, save_name_b)
            


if __name__ == '__main__':
    main()