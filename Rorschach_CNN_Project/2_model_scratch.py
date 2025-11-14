import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --- 1. Define Hyperparameters and Paths (MODIFIED FOR RORSCHACH TASK) ---
IMAGE_SIZE = (64, 64)  # Small resolution to limit model complexity
BATCH_SIZE = 32
EPOCHS = 50

# **CRITICAL CHANGE:** NUM_CLASSES now represents the number of distinct animal responses 
# humans are likely to give (e.g., Bat, Butterfly, Dog, Bear, Spider).
NUM_CLASSES = 5 
DATASET_PATH = 'path/to/your/inkblot_images_with_animal_labels/' 

# --- 2. Data Preprocessing and Augmentation ---
# NOTE: Augmentation may need to be limited for Rorschach images to preserve symmetry.
train_datagen = ImageDataGenerator(
    rescale=1./255,         
    rotation_range=5,       # Reduced rotation range to maintain symmetry
    width_shift_range=0.05, 
    height_shift_range=0.05,
    shear_range=0.05,       
    zoom_range=0.05,        
    # **CRITICAL CHANGE:** horizontal_flip MUST be set to False if symmetry is vital.
    horizontal_flip=False, 
    validation_split=0.2    
)

# Load data from directory. Each folder name in DATASET_PATH must be an animal response (e.g., /Bat, /Butterfly).
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Still using categorical for multiple discrete classes
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- 3. Define the Response Predictor CNN Model ---
def build_response_predictor_cnn(input_shape, num_classes):
    # This structure is identical to your animal classifier, but its purpose is now different.
    # It learns to recognize the features in an inkblot that map to a specific animal response.
    model = Sequential([
        # 1st Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        # 2nd Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        
        # Fully Connected Layer 
        Dense(128, activation='relu'),
        
        # Regularization 
        Dropout(0.5), 
        
        # Output Layer: predicts the probability of each animal response class
        Dense(num_classes, activation='softmax') 
    ])
    return model

model = build_response_predictor_cnn(IMAGE_SIZE + (3,), NUM_CLASSES)

# --- 4. Compile and Train the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Appropriate loss for multi-class prediction
    metrics=['accuracy']
)

print(model.summary())

# Train the model (This step must run successfully before Phase Two can begin)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. Plotting Results (for analysis) ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Once this model achieves high accuracy, it is saved and used as the "Perceptual Loss" evaluator
# in the GAN's training loop (Phase Two of the architecture).