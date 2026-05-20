# Plant Disease Detection

This project trains a convolutional neural network (CNN) to classify plant leaf images and identify disease types. It uses TensorFlow/Keras and the PlantVillage dataset.

## Project structure

- `app.py` - Application entry point (likely for inference or serving predictions).
- `train_model.py` - Training script for building and saving the CNN model.
- `Dataset/PlantVillage/` - Image dataset organized by class folders.
- `models/plant_disease_model.h5` - Saved trained model.

## Requirements

- Python 3.8+
- TensorFlow
- Matplotlib

## Usage

1. Install dependencies:

```bash
pip install tensorflow matplotlib
```

2. Run training:

```bash
python train_model.py
```

3. The model will be saved to `models/plant_disease_model.h5`.

## Notes

- The dataset folder must contain class subdirectories under `Dataset/PlantVillage/`.
- The training script performs data augmentation and validation split.
