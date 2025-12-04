import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Basically this is the brain for train.py
def build_model_scratch(num_classes):
    print("--- Building model from scratch (Model A)")
    
    # We use 224x224 to match your data_pipeline.py
    # We use 3 color channels (RGB)
    # We use 224 as the shape because the model transfer that we will use uses 224, instead of 64 that we were trying before
    inputs = keras.Input(shape=(224, 224, 3))

    # We start with 32 filters to look for simple edges
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # We increase to 64 filters to find shapes (circles, corners)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # We increase to 128 filters to find complex objects (eyes, ears)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten the 3D maps into a 1D list of numbers
    x = layers.Flatten()(x)
    
    # Dense layer for "thinking"
    x = layers.Dense(128, activation='relu')(x)
    
    # Dropout to prevent memorization (overfitting)
    x = layers.Dropout(0.5)(x)
    
    # Final Output Layer (10 neurons for 10 animals)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Combine inputs and outputs into a Model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model