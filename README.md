# Alzheimer's MRI Classification Project (CS 184A)

A deep learning project for classifying Alzheimer's disease stages from brain MRI scans using Vision Transformer (ViT) and EfficientNet models.

## Overview

This project uses transfer learning with pre-trained models to classify MRI brain scans into 4 categories:
- **MildDemented**
- **ModerateDemented**  
- **NonDemented**
- **VeryMildDemented**

## Getting Started

### Option 1: Running demo.ipynb (Recommended)

This option uses pre-trained weights for quick inference without training.

1. **Download the weights** from [Google Drive](https://drive.google.com/file/d/19i9Un-N3z9af3r3jse9m14KwAAkUtuRQ/view?usp=sharing)
2. **Extract the zip** and move the `vit-best` and `effnet-best` folders to the project root directory
3. **Set up the environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
4. **Run the notebook** - Execute cells sequentially to see model predictions with confidence scores

### Option 2: Running project.ipynb (Full Training)

This option trains the models from scratch using Google Colab.

1. Open `project.ipynb` in **Google Colab**
2. Set runtime to **GPU** (Runtime → Change runtime type → GPU)
3. Upload your `kaggle.json` file for dataset access
4. In Cell 9, ensure dataset is set to `"OriginalDataset"` for faster training
5. Run all cells sequentially

## Project Structure

```
├── demo.ipynb          # Quick demo with pre-trained weights
├── project.ipynb       # Full training notebook
├── requirements.txt    # Python dependencies
├── sampledata/         # Dataset directory
│   ├── OriginalDataset/
├── vit-best/           # Pre-trained ViT weights
└── effnet-best/        # Pre-trained EfficientNet weights
```

## Models

- **Vision Transformer (ViT)**: `google/vit-base-patch16-224`
- **EfficientNet**: `google/efficientnet-b0`

Both models use K-Fold cross-validation for robust training evaluation.

## Notes

- Training time is moderately long due to dataset size constraints
- Pre-trained weights are provided to skip training and demonstrate inference
- The prediction cells display images with true labels, predicted labels, and confidence percentages
