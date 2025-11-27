import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import shutil



''' #SCRIPT FOR DATASET CLEANING (ISOLATED CELLS)
base_dir = 'dataset'
classes = os.listdir(base_dir)

#Cleaning the dataset folder such that subfolders contain only images 

for cls in classes:
    cls_dir = os.path.join(base_dir, cls)
    target_dir = os.path.join(base_dir, cls[3:] + '_flat')
    os.makedirs(target_dir, exist_ok=True)
    
    for sub in os.listdir(cls_dir):
        sub_path = os.path.join(cls_dir, sub)
        if 'CROPPED' in sub:
            for file in os.listdir(sub_path):
                if file.endswith('.bmp'):
                    src_path = os.path.join(sub_path, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.copy(src_path, dst_path)
                    
#Verifying the number of images in each cleaned folder                     
                    
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and folder.endswith('_flat'):
        images = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]
        print(f"{folder}: {len(images)} images")
'''

IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'dataset'


train_ds = tf.keras.preprocessing.image_dataset_from_directory( #train dataset
    DATA_DIR,
    validation_split=0.3,     
    subset='training',  
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)


val_test_ds = tf.keras.preprocessing.image_dataset_from_directory( #val/test dataset (later splitted in half)
    DATA_DIR,
    validation_split=0.3,
    subset='validation',
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_size = int(0.5 * len(val_test_ds))  #halving validation dataset for the test set
val_ds = val_test_ds.take(val_size) #takes first half of val_size
test_ds = val_test_ds.skip(val_size) #takes the second half of val_size


def data_aug(): #creates image augmentation layers
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(layers.RandomFlip('horizontal'))
    data_augmentation.add(layers.RandomRotation(0.05))
    data_augmentation.add(layers.RandomZoom(0.15))
    data_augmentation.add(layers.RandomContrast(0.2))
    data_augmentation.add(layers.RandomBrightness(0.2))

    return data_augmentation



def img_augmentation(images): #inputs an image into the data_aug layers
    aug_layer = data_aug()        
    return aug_layer(images)


''' #VISUALIZATION OF IMAGES AFTER AUGMENTATION
for image, label in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(image)
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title(i)
        plt.axis("off")
plt.suptitle("AUGMENTED SAMPLES", fontsize=16)
plt.show()
'''

print(f"Train classes  {train_ds.class_names}")
print(f"Val classes  {val_test_ds.class_names}")
print(f"Test classes  {val_test_ds.class_names}")