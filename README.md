# Deep Learning Assignments

## Manikanta Rajulapati
## 700762001

### **Overview**
This repository contains multiple Jupyter Notebooks related to deep learning concepts, including convolution operations, CNN architectures, cloud computing, and model comparisons.

---

### **1. Question_1.ipynb - Cloud Computing for Deep Learning**
#### **(a) Elasticity and Scalability**
- Defines elasticity and scalability in the context of deep learning and cloud computing.
- Explains key differences and use cases.

#### **(b) Comparison of AWS SageMaker, Google Vertex AI, and Microsoft Azure ML**
- Compares deep learning capabilities of three cloud platforms.
- Highlights key features, integrations, and cost structures.
  
---
### **2. Question_2.ipynb - Convolution Operations with Different Parameters**
- Implements convolution using NumPy and TensorFlow/Keras.
- Applies different stride and padding values.
- Outputs feature maps for:
  - Stride = 1, Padding = 'VALID'
  - Stride = 1, Padding = 'SAME'
  - Stride = 2, Padding = 'VALID'
  - Stride = 2, Padding = 'SAME'

**Example Code Snippet:**
```python
import numpy as np
import tensorflow as tf

input_matrix = np.random.rand(5,5)
kernel = np.random.rand(3,3)

conv_result = tf.nn.conv2d(
    input=tf.expand_dims(tf.expand_dims(input_matrix, axis=0), axis=-1),
    filters=tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1),
    strides=[1,1,1,1],
    padding='VALID'
)
print(conv_result.numpy())
```

---

### **3. Question_3.ipynb - CNN Feature Extraction with Filters and Pooling**
#### **Task 1: Edge Detection Using Sobel Filter**
- Loads a grayscale image.
- Applies Sobel filters for edge detection in the X and Y directions.
- Displays original and filtered images.

#### **Task 2: Max Pooling and Average Pooling**
- Generates a 4x4 random matrix.
- Applies 2x2 Max Pooling and Average Pooling.
- Outputs the original matrix and pooled matrices.

**Example Code Snippet:**
```python
import cv2
import numpy as np

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
```

---

### **4. Question_4.ipynb - Implementing and Comparing CNN Architectures**
#### **Task 1: AlexNet Implementation**
- Builds a simplified AlexNet architecture using TensorFlow/Keras.
- Includes convolutional, pooling, dropout, and fully connected layers.
- Displays model summary.

#### **Task 2: ResNet-like Model with Residual Blocks**
- Defines a residual block with skip connections.
- Builds a simple ResNet model with an initial convolutional layer and two residual blocks.
- Displays model summary.

**Example Code Snippet:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
    MaxPooling2D((3, 3), strides=2),
    Conv2D(256, (5, 5), activation='relu', padding="same"),
    MaxPooling2D((3, 3), strides=2),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()
```

---

## **Requirements**
To run these notebooks, install the required dependencies:
```bash
pip install numpy tensorflow keras matplotlib opencv-python
```

## **Usage**
Run each notebook in Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook
```

