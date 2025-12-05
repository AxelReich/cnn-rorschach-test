# CNN Rorschach Test: AI Pareidolia Experiment

An experimental deep learning project that investigates AI pareidolia by training convolutional neural networks on animal images and then testing what patterns they perceive in ambiguous Rorschach inkblots.

## Overview

This project explores how different neural network architectures—a naive model trained from scratch versus an expert model using transfer learning—interpret ambiguous visual stimuli. Similar to psychological Rorschach tests, we analyze what "shapes" or "patterns" the AI models see in inkblot images.

## Project Structure

```
cnn-rorschach-test/
├── Rorschach_CNN_Project/
│   ├── data/
│   │   ├── animals-10/          # Training dataset (10 animal classes)
│   │   └── rorschach_blots/     # Test images (10 Rorschach inkblots)
│   ├── saved_models/            # Trained model weights
│   ├── experiment_results/      # GradCAM visualizations
│   ├── data_pipeline.py         # Data loading and preprocessing
│   ├── model_scratch.py         # Model A: CNN built from scratch
│   ├── model_transfer.py        # Model B: Transfer learning with MobileNetV2
│   ├── train.py                 # Main training script
│   ├── predict_and_visualize.py # Inference and GradCAM visualization
│   ├── analysis_notebook.ipynb  # Results visualization notebook
│   └── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

## Dataset

### Training Data: Animals-10
- **Source**: Animals-10 dataset
- **Classes**: 10 animal categories
  - Butterfly (2,112 images)
  - Cat (1,668 images)
  - Chicken (3,098 images)
  - Cow (1,866 images)
  - Dog (4,863 images)
  - Elephant (1,446 images)
  - Horse (2,623 images)
  - Sheep (1,820 images)
  - Spider (4,821 images)
  - Squirrel (1,862 images)
- **Total**: ~25,000+ images
- **Split**: 80% training, 20% validation

### Test Data: Rorschach Inkblots
- 10 Rorschach inkblot images (Rorschach1.webp through Rorschach10.webp)
- Used to test what patterns/models the AI "sees" in ambiguous shapes

## Models

### Model A: Naive (Built from Scratch)
- **Architecture**: Simple CNN
  - 3 Convolutional layers (32 → 64 → 128 filters)
  - MaxPooling after each conv layer
  - Dense layer (128 neurons) with 50% dropout
  - Output layer (10 classes)
- **Training**: 15 epochs
- **Approach**: Learns features from scratch

### Model B: Expert (Transfer Learning)
- **Base Model**: MobileNetV2 (ImageNet pretrained weights)
- **Architecture**:
  - Frozen MobileNetV2 base (initial training)
  - Global Average Pooling
  - Dense layer with dropout (20%)
  - Output layer (10 classes)
- **Training Strategy**:
  - **Phase 1**: Train new head with frozen base (10 epochs, lr=0.0001)
  - **Phase 2**: Fine-tune top layers (10 epochs, lr=1e-5)
- **Approach**: Leverages pretrained knowledge

## Model Performance

After training on the Animals-10 dataset, the models achieved the following validation accuracy:

| Model | Architecture | Validation Accuracy |
|-------|-------------|---------------------|
| **Model A** | CNN from Scratch | **67%** |
| **Model B** | MobileNetV2 Transfer Learning (Fine-tuned) | **96%** |

### Performance Analysis

- **Model A (Naive)**: Achieved 67% accuracy, demonstrating that a simple CNN can learn basic features but struggles with the complexity of animal classification without pretrained knowledge.

- **Model B (Expert)**: Achieved 96% accuracy, showcasing the power of transfer learning. By leveraging MobileNetV2's pretrained ImageNet weights and fine-tuning, the model successfully adapts to the animal classification task.

The significant performance gap (29%) highlights the importance of transfer learning when working with limited datasets and complex visual recognition tasks.

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd cnn-rorschach-test
```

2. **Install dependencies**:
```bash
cd Rorschach_CNN_Project
pip install -r requirements.txt
```

**Required packages**:
- tensorflow
- matplotlib
- numpy
- scipy

## Usage

### 1. Training the Models

Train both Model A and Model B:
```bash
cd Rorschach_CNN_Project
python train.py
```

This will:
- Load and preprocess the Animals-10 dataset
- Train Model A from scratch (15 epochs)
- Train Model B using transfer learning (20 epochs total)
- Save models to `saved_models/`
- Generate training history plots

**Note**: Ensure the `data/animals-10/` directory contains the training images organized by class folders.

### 2. Generating Predictions and Visualizations

After training, run predictions on Rorschach inkblots:
```bash
python predict_and_visualize.py
```

This will:
- Load the trained models
- Process each Rorschach inkblot image
- Generate GradCAM heatmaps showing what each model focuses on
- Save visualizations to `experiment_results/`

### 3. Viewing Results

Open the Jupyter notebook to visualize comparisons:
```bash
jupyter notebook analysis_notebook.ipynb
```

Or view the saved images directly in `experiment_results/`:
- `Rorschach[N]_ModelA.jpg` - Model A's visualization
- `Rorschach[N]_ModelB.jpg` - Model B's visualization

## Results

### Classification Performance
- **Model A**: 67% validation accuracy on Animals-10 dataset
- **Model B**: 96% validation accuracy on Animals-10 dataset

### Rorschach Test Visualizations

The project generates GradCAM visualizations that highlight the regions of Rorschach inkblots that each model focuses on when making predictions. These heatmaps reveal:

- **Model A (Naive, 67% accuracy)**: Tends to focus on simpler, more general patterns due to its limited feature learning capacity
- **Model B (Expert, 96% accuracy)**: Focuses on more complex, learned features from ImageNet, demonstrating more sophisticated pattern recognition

The visualizations allow you to compare how different learning approaches (and their corresponding performance levels) interpret ambiguous visual stimuli. The performance gap between models is reflected in their different interpretations of the Rorschach inkblots.

## Key Features

- **Data Augmentation**: Rotation, shifts, shear, zoom, and horizontal flips for training robustness
- **Transfer Learning**: Leverages MobileNetV2 pretrained on ImageNet
- **GradCAM Visualization**: Understands what the models "see" in ambiguous images
- **Comparative Analysis**: Side-by-side comparison of naive vs. expert models

## Configuration

Key parameters can be modified in the respective files:

- **`data_pipeline.py`**: Image size (224×224), batch size (32), augmentation settings
- **`train.py`**: Number of epochs for each model
- **`predict_and_visualize.py`**: Model paths, output directories

## License

MIT License - see LICENSE file for details

## Author

Axel Reich (2025)
Tanner Rohloff (2025) 

## Notes

- The models are saved in TensorFlow's `.keras` format
- Training history plots are automatically saved to `saved_models/`
- Ensure sufficient disk space for the dataset (~GB range)
- GPU recommended for faster training, but CPU will work

