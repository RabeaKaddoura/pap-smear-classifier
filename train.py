import tensorflow as tf
from tensorflow.keras import layers
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



def img_augmentation(images): #inputs an image into the data_aug layers, returns an augmented image
    aug_layer = data_aug()        
    return aug_layer(images)


def plt_aug_img(): #Visualization of images after augmentation
    for image, label in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            aug_img = img_augmentation(image)
            plt.imshow(aug_img[0].numpy().astype("uint8"))
            plt.title(i)
            plt.axis("off")
    plt.suptitle("AUGMENTED SAMPLES", fontsize=16)
    plt.show()
    



print(f"Train classes  {train_ds.class_names}")
print(f"Val classes  {val_test_ds.class_names}")
print(f"Test classes  {val_test_ds.class_names}")

#print(train_ds)




def pap_smear_model(dropout_rate): #model structure
    
    base_model = tf.keras.applications.EfficientNetB0 (
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        name="efficientnetb0",
    )
    
    base_model.trainable = False #freezing layers
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug()(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x) #automatic preprocessing of the input image (normalization, etc.)

    x = base_model(x, training = False) #taking image as input to EfficientNet network using the pre-trained weights (after unfreeze)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(5, "softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model


model = pap_smear_model(0.6) #dropout rate

initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( #applying learning rate decay
    initial_learning_rate,
    decay_steps=10,
    decay_rate=0.95,
    staircase=True)

model.compile(
    optimizer= tf.keras.optimizers.Adam(lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


initial_epochs = 25

history = model.fit(train_ds, validation_data= val_ds, epochs=initial_epochs) #training the model with freeze EfficientNetB0 layers



#-------------Hypertuning----------------


base_model = model.layers[2] #extracting the pre-trained model (EfficientNetB0)
base_model.trainable = True
fine_tune_from = 90 #EfficientNetB0 has a total of 237 layers

for layer in base_model.layers[:fine_tune_from]: #freezing layers before fine_tune_from
    layer.trainable = False

fine_tune_loss = 'sparse_categorical_crossentropy'


fine_tune_learning_rate = 0.1 * initial_learning_rate


fine_tune_optimizer = tf.keras.optimizers.Adam(
    learning_rate=fine_tune_learning_rate,
)
    

model.compile(
    optimizer=fine_tune_optimizer,
    loss=fine_tune_loss,
    metrics=['accuracy']
    )

fine_tune_epochs = initial_epochs + 28

fine_tune_history = model.fit( #training the fine tuned model with unfreezed layers from EfficientNetB0
    train_ds,
    validation_data= val_ds,
    initial_epoch=history.epoch[-1],
    epochs = fine_tune_epochs
)


test_loss, test_acc = model.evaluate(test_ds) #testing the model
print(f"Test Accuracy: {test_acc:.4f}")





plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show() 


model.save("pap_smear_model_final.keras")
