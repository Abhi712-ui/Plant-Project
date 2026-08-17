# Crop Disease Classification

An end to end deep learning system that identifies plant diseases based on photographs of leaves.\
Upload an image of a leaf and the model returns the crop species, the disease present (or if no disease is present, a healthy determination), and a confidence score.\
[Live Demo](https://plant-project-4pmiukf5c7lyix79q2vvaf.streamlit.app/)\
![app demo](app_demo.jpg)

## Results

| Metric | Value |
| --- | --- |
| Validation accuracy | 98.7% |
| Classes | 38 (14 crop species, in both healthy and diseased states) |
| Training images | ~70,000 |
| Architecture | ResNet18, fine-tuned through transfer learning |

## Diagnosing and Correcting Overfitting

- The initial model was trained without augmentation, and received around 95.6 percent validation accuracy after one epoch, yet declined to 94.7 percent after the second epoch, while training accuracy rose from 92% to 96.4%.
- Validation loss also increased, even as training loss fell.
- This divergence that was observed between training and validation performance was a clear indicator of overfitting - the model was memorizing the training set instead of generalizing.

### Solution

#### Data Augmentation on the Training Set

- Random horizontal flips and rotations upt to 15 degrees.
- Validation images were left unaltered, so that evaluation remained a measure of performance on unmodified data.

#### Validation based checkpointing

- Weights were saved whenever validation accuracy reached a new best, instead of only at the end of training
- In the baseline training, the best model was produced after one epoch, but was overwritten by weaker weights in epoch two

