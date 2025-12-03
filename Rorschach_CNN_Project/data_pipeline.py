# Open this file: 1_data_pipeline.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
# Set key variables that we'll use in all our files
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32  # How many images to load at a time

# Get the path to your data directory
# This assumes your 'data' folder is in the same place as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'animals-10')

def get_data_generators():
    """
    Creates and configures the training and validation data generators.
    """
    
    # --- 1. The Training Generator (with Data Augmentation) ---
    # This creates "fake" new images to help the model generalize
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Normalize pixel values from 0-255 to 0.0-1.0
        rotation_range=20,         # Randomly rotate images
        width_shift_range=0.2,     # Randomly shift images horizontally
        height_shift_range=0.2,    # Randomly shift images vertically
        shear_range=0.2,           # "Slant" the image
        zoom_range=0.2,            # Randomly zoom in
        horizontal_flip=True,      # Randomly flip images horizontally
        validation_split=0.2       # Automatically reserve 20% of data for validation
    )

    # --- 2. The Validation Generator (NO Augmentation) ---
    # We only normalize the validation data. We don't augment it
    # because we want to test the model on the *original*, untouched images.
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2       # Must be the same split as above
    )

    # --- 3. Create the Generator Objects ---
    # These "flow" data from your directory, applying the rules above.
    
    print(f"Loading training images from: {DATA_DIR}")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,      # Resize all images to 224x224
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # We have >2 classes
        subset='training',         # Label this as the training set
        shuffle=True
    )

    print(f"Loading validation images from: {DATA_DIR}")
    validation_generator = validation_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',     # Label this as the validation set
        shuffle=False            # No need to shuffle validation data
    )
    
    # Get the number of classes (should be 10)
    num_classes = train_generator.num_classes
    print(f"Found {num_classes} classes.")
    print(f"Class indices: {train_generator.class_indices}")

    return train_generator, validation_generator, num_classes

# This "if __name__ == '__main__':" block lets you
# run this file directly from the terminal to test it.
if __name__ == '__main__':
    print("Testing the data pipeline...")
    train_gen, val_gen, classes = get_data_generators()
    print(f"\nSuccessfully created generators.")
    print(f"Number of classes detected: {classes}")
    
    # Optional: Grab one batch of images and print its shape
    images, labels = next(train_gen)
    print(f"Shape of one image batch (Batch Size, H, W, Channels): {images.shape}")
    print(f"Shape of one label batch (Batch Size, Num Classes): {labels.shape}")