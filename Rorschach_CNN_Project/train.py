import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# --- Import our custom modules ---
# We use 'as' to give them short nicknames
import data_pipeline as data    # Refers to data_pipeline.py
import model_scratch as scratch # Refers to model_scratch.py
import model_transfer as transfer # Refers to model_transfer.py

# --- Configuration ---
# You can adjust these numbers. 
# 15 epochs is usually enough for the scratch model to start learning without taking all day.
EPOCHS_SCRATCH = 15      
EPOCHS_TRANSFER_HEAD = 10 # Initial training for transfer model
EPOCHS_FINE_TUNE = 10    # Fine-tuning for transfer model
SAVE_DIR = 'saved_models'

# Create directory to save models if it doesn't exist
if not os.path.exists(SAVE_DIR):    
    os.makedirs(SAVE_DIR)

def plot_history(history, title):
    """
    Helper function to plot accuracy graphs.
    This saves a .png image so you can SEE if your model learned.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title}: Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title}: Loss')
    
    # Save the plot
    plt.savefig(f"{SAVE_DIR}/{title.replace(' ', '_')}_plot.png")
    print(f"Graph saved as {title.replace(' ', '_')}_plot.png")

def main():
    # 1. Get the Data
    print("\n=== STEP 1: Loading Data ===")
    train_gen, val_gen, num_classes = data.get_data_generators()
    
    # ---------------------------------------------------------
    # 2. Train Model A (From Scratch)
    # ---------------------------------------------------------
    print("\n=== STEP 2: Training Model A (From Scratch) ===")
    model_a = scratch.build_model_scratch(num_classes)
    
    history_a = model_a.fit(
        train_gen,
        epochs=EPOCHS_SCRATCH,
        validation_data=val_gen
    )
    
    # Save Model A
    # The .keras format is the modern standard for TensorFlow
    save_path_a = os.path.join(SAVE_DIR, 'model_A_scratch.keras')
    model_a.save(save_path_a)
    print(f"Model A saved to: {save_path_a}")
    
    plot_history(history_a, "Model A Scratch")

    # ---------------------------------------------------------
    # 3. Train Model B (Transfer Learning)
    # ---------------------------------------------------------
    print("\n=== STEP 3: Training Model B (Transfer Learning) ===")
    
    # Part 3a: Initial Training (Head Only)
    print("--- Phase 1: Training Head ---")
    model_b, base_model = transfer.build_model_transfer(num_classes)
    
    history_b1 = model_b.fit(
        train_gen,
        epochs=EPOCHS_TRANSFER_HEAD,
        validation_data=val_gen
    )
    
    # Part 3b: Fine-Tuning (The Expert Phase)
    print("--- Phase 2: Fine-Tuning Expert Layers ---")
    model_b = transfer.fine_tune_model(model_b, base_model)
    
    # Train again, picking up where we left off
    # We add the new epochs to the old ones so the graph looks right
    total_epochs = EPOCHS_TRANSFER_HEAD + EPOCHS_FINE_TUNE
    
    history_b2 = model_b.fit(
        train_gen,
        epochs=total_epochs,
        initial_epoch=history_b1.epoch[-1], # Start from last epoch
        validation_data=val_gen
    )
    
    # Save Model B
    save_path_b = os.path.join(SAVE_DIR, 'model_B_transfer.keras')
    model_b.save(save_path_b)
    print(f"Model B saved to: {save_path_b}")
    
    # We plot the fine-tuning history
    plot_history(history_b2, "Model B FineTuned")

    print("\n=== ALL TRAINING COMPLETE ===")
    print("You are now ready to run 'predict_and_visualize.py'")

if __name__ == '__main__':
    main()