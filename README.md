# Image Data Augmentation with TensorFlow (MNIST)

##  Overview
This project demonstrates how image data augmentation improves model generalization by reducing overfitting.
Using TensorFlow's image augmentation APIs, multiple transformations are applied both to a single image and dynamically during model training.

---

##  Augmentation Techniques Implemented
- Horizontal flipping
- Grayscale conversion
- Brightness adjustment
- Saturation adjustment
- Rotation
- Random cropping with padding
- Random brightness

All augmentations are applied **on-the-fly** using `tf.data` pipelines.

---

##  Dataset
- MNIST handwritten digits
- Training subset: 2,048 samples
- Validation: full test set

---

##  Model Architecture
- Fully Connected Neural Network
- 2 hidden layers (4096 units each)
- ReLU activation
- Optimizer: Adam

---

##  Results
| Model | Validation Accuracy |
|------|---------------------|
| Without Augmentation | ~94% |
| With Augmentation | **~95%** |

Data augmentation improved generalization and reduced overfitting despite slower convergence.

---

##  How to Run
```bash
pip install -r requirements.txt
python src/train.py
