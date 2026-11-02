# Plant Disease Detection with Transfer Learning

Classifying 15 plant leaf diseases from images by comparing four pretrained CNNs (MobileNet, DenseNet121, VGG16, InceptionV3) on the PlantVillage dataset.

**Best model: MobileNet, 93.8% validation accuracy and 93.9% F1-score**, while also being the lightest and fastest of the four.

![Model comparison](images/model_comparison.png)

## Overview

Crop diseases are often spotted too late by eye. This project tests how well off-the-shelf ImageNet models can recognise diseased leaves when used as frozen feature extractors with a small custom classifier on top. It was built for a Computational Biology course at Bina Nusantara University.

## Dataset

- **Source:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (Kaggle)
- **Size:** 20,638 leaf images across 15 classes (pepper, potato and tomato, healthy and diseased)
- **Split:** 80% training (16,516 images) and 20% validation (4,122 images)

The dataset is not included in this repo. Download it and point `train_datset_dir` in the notebook to the `PlantVillage` folder.

## Method

1. **Preprocessing:** resize to 224×224, rescale pixels to [0, 1].
2. **Augmentation:** random rotation (±20°) and horizontal flips on the fly.
3. **Transfer learning:** each pretrained base is frozen, then topped with
   `Flatten → Dense(256, ReLU) → Dense(64, ReLU) → Dense(15, Softmax)`.
4. **Training:** Adam optimizer, categorical cross-entropy, batch size 32, 10 epochs.
5. **Evaluation:** accuracy, precision, recall and F1-score (custom Keras metric) on the validation set.

## Results

| Model | Accuracy | F1-score | Precision | Recall |
|---|---|---|---|---|
| **MobileNet** | **93.83%** | **93.90%** | **94.16%** | **93.64%** |
| DenseNet121 | 93.37% | 93.47% | 93.79% | 93.15% |
| VGG16 | 90.12% | 90.40% | 91.02% | 89.78% |
| InceptionV3 | 87.65% | 88.00% | 89.53% | 86.53% |

**Takeaways**

- MobileNet matched or beat much larger models, making it the best choice for a lightweight, deployable app (e.g. on a phone in the field).
- DenseNet121 was a close second and had the lowest validation loss.
- All models scored 3–4 points lower on validation than on training, a sign of mild overfitting.

## Limitations and Next Steps

- **No separate test set.** The validation set was used for both monitoring and the final scores, so the numbers may be slightly optimistic. A held-out test split would give a cleaner estimate.
- **Loss setting.** The loss uses `from_logits=True` while the output layer already applies softmax; switching to `from_logits=False` is the correct setting and may improve training.
- **Lab images.** PlantVillage photos have plain backgrounds; accuracy on real field photos would likely be lower.
- **Next steps:** fine-tune the top layers of the best base model, add early stopping, and report a per-class confusion matrix.

## Tech Stack

Python · TensorFlow / Keras · NumPy · pandas · Matplotlib

## How to Run

```bash
pip install tensorflow numpy pandas matplotlib jupyter
jupyter notebook "Computational Biology.ipynb"
```

## Author

**Darel Yeshe**, Data Science graduate, Bina Nusantara University · [GitHub](https://github.com/Darel-Yeshe)
