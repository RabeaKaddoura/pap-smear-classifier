# Overview
A machine learning model that classifies individual cells taken from a Pap Smear into one of 5 classes: **Dyskeratotic (Abnormal), Koilocytotic (Abnormal), Metaplastic (Benign), Parabasal (Normal), and Superficial-Intermediate (Normal)**. The model is based on the pre-trained convolutional neural network **EfficientNetB0**, which has a total of 237 layers, and was chosen among other pre-trained networks due to its well-known high accuracy for recognizing medical imaging diagnosis problems (Samma & Hamad, 2024).  

---

# Dataset
The dataset comprises **4049 isolated cell images** that are taken from Pap Smear slides (Cropped), of different sizes. For that, and for the fact that EfficientNetB0 requires a specific input size of **224x224x3**, a pre-processing step of resizing each image into **224x224x3** was done. Each isolated cell image belongs to one of the 5 classes: **Dyskeratotic (Abnormal), Koilocytotic (Abnormal), Metaplastic (Benign), Parabasal (Normal), and Superficial-Intermediate (Normal)**.

---

# Model
The model starts by resizing each input image to **224x224x3**, which is then taken to another pre-processing step: **Augmentation**. Augmentation is necessary in this case since I was limited in the number of images of 4049. For that, I added an **augmentation layer** that applies different augmentation effects such as random rotation of the image, random zoom, random contrast, etc. This dramatically increased the amount of training data, which in turn greatly improved the accuracy of the model.  

After the preprocessing step, I started with **EfficientNetB0** layer, which was initially frozen so that only the dense layer’s weights are trained at first. After the dense layer weights were trained, I reduced the number of frozen layers to 90. From there, the model was trained on a total of **53 epochs**, **batch of 32**, a **70/15/15 split**, and was fine-tuned over multiple iterations to include **Adam optimizer**, a **dropout rate of 0.6**, and an **exponential learning rate decay**.

![PAP Model Architecture](PAP_MODEL_ARCHITETCURE.png)
---

# Results
The model achieved a satisfactory **test accuracy of 98.35%** with minimal bias/variance:

![Accuracy Graph](Accuracy_Graph.png)
