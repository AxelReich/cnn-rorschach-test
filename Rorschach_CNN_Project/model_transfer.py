import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

def build_model_transfer(num_classes):
    print("--- Building Transfer Learning Model (Model B) ---")

    # 1. Define Input Shape 
    # Must match the 224x224 size from your data pipeline!
    IMG_SHAPE = (224, 224, 3)

    # 2. Load the Pre-trained Base Model (The "Expert Brain")
    # We use MobileNetV2 because it's fast and accurate.
    # weights='imagenet' loads the "world knowledge" it learned from 1.4M photos.
    # include_top=False chops off the original classifier so we can add our own.
    base_model = MobileNetV2(input_shape=IMG_SHAPE,
                             include_top=False, 
                             weights='imagenet')

    # 3. Freeze the Base
    # We lock the "Expert Brain" so our training doesn't destroy its knowledge.
    base_model.trainable = False

    # 4. Build the Architecture
    inputs = keras.Input(shape=IMG_SHAPE)
    
    # We pass the image through the Expert Brain
    # training=False ensures the internal "BatchNormalization" layers stay in inference mode
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
    # We use a standard learning rate (0.0001) for this first phase
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Return BOTH the full model AND the base_model (we need the base for fine-tuning later)
    return model, base_model

def fine_tune_model(model, base_model):
    """
    This function unfreezes the top layers of the Expert Brain.
    It turns the model from a "Generalist" into a "Specialist" on your animals.
    """
    print("--- Unfreezing top layers for Fine-Tuning ---")
    
    # 1. Unfreeze the base model
    base_model.trainable = True
    
    # 2. Refreeze the bottom layers (Keep the basic shapes/edges frozen)
    # MobileNetV2 has 154 layers. We will only train the top 54.
    fine_tune_at = 100
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # 3. Re-compile with a VERY LOW learning rate
    # This is critical! If the rate is too high, the model "forgets" everything.
    model.compile(loss='categorical_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5), # 10x smaller rate
                  metrics=['accuracy'])
    
    return model