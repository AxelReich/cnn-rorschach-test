import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

def build_model_transfer(num_classes):
    print("--- Building Transfer Learning Model (Model B) ---")

    IMG_SHAPE = (224, 224, 3)

    # Load the Pre-trained Base Model (The "Expert Brain"), use MobileNetV2 because it's fast and accurate.
    base_model = MobileNetV2(input_shape=IMG_SHAPE,
                             include_top=False, 
                             weights='imagenet')

    # We lock the "Expert Brain" so our training doesn't destroy its knowledge.
    base_model.trainable = False

    # Build the Architecture
    inputs = keras.Input(shape=IMG_SHAPE)
    
    # We pass the image through the Expert Brain
    x = base_model(inputs, training=False)
    
    # Convert the 3D features into a 1D vector
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add a dropout layer to prevent overfitting
    x = layers.Dropout(0.2)(x)
    
    # Final Output Layer (10 neurons for your 10 animals)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Combine into a Model
    model = keras.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model, base_model

def fine_tune_model(model, base_model):
    """
    This function unfreezes the top layers of the Expert Brain.
    It turns the model from a "Generalist" into a "Specialist" on your animals.
    """
    
    # 1. Unfreeze the base model
    base_model.trainable = True
    
    fine_tune_at = 100
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(loss='categorical_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
                  metrics=['accuracy'])
    
    return model