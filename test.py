import tensorflow as tf
from tensorflow.keras import  *
import os

model = tf.keras.models.load_model("pap_smear_model_final.keras")
f_path = "test_dataset"


for img in os.listdir(f_path):
    img_path = os.path.join(f_path, img)
    img = tf.keras.utils.load_img(img_path, target_size=(224,224))
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, axis = 0)

    prediction = model.predict(img)
    
    class_names = ['Dyskeratotic_flat', 'Koilocytotic_flat', 'Metaplastic_flat', 'Parabasal_flat', 'Superficial-Intermediate_flat']
    cls = class_names[prediction.argmax()]
    print(f"IN: {img_path} ---> PRED: {cls[0:-5]}")