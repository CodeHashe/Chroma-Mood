#Importing Libraries
import random 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks 
from  tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from keras import regularizers
import mlflow
import mlflow.tensorflow
mlflow.tensorflow.autolog()
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

#Declaring Global Variables
dataset_dir=('./fer2013')
img_height=48
img_width=48
img_channels=1 #RGB OR GREYSCALE WE USING GREYSCALE
batch_size=64
epoch=60
learn_rate=0.0001
seed=12
#Classes
classes=['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
num_classes=len(classes)
#Saving Path for the model
model_save_dir='emotionsvideomodel.h5'



#Data Preprocessing and Augmentation

train_dir=os.path.join(dataset_dir,'train')
test_dir=os.path.join(dataset_dir,'test')

train_data= ImageDataGenerator(
                            width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            horizontal_flip = True,
                            rescale = 1./255,
                            validation_split = 0.2
                        )
test_data  = ImageDataGenerator(rescale=1./255,validation_split = 0.2)

 # Training generator
train_generator = train_data.flow_from_directory(
train_dir,
target_size = (img_height,img_width),
color_mode  = "grayscale",
batch_size  = batch_size,
class_mode  = "categorical",
seed        = 42, #To ensure sample stability between sessions.
subset="training"   
)
 # Validation generator (from same training dir using validation_split)
    
validation_generator=train_data.flow_from_directory(
train_dir,
target_size = (img_height,img_width),
color_mode = "grayscale",
batch_size = batch_size,
class_mode="categorical",
seed       = 42 ,      #To ensure sample stability between sessions.
subset="validation"     
)
test_generator = test_data.flow_from_directory(
    test_dir,
    target_size = (img_height,img_width),
    color_mode = "grayscale",
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle    = False,
)




#Building CNN

model = models.Sequential([
    # Block 1 - Input Layer
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(img_height,img_width,img_channels)),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(128, (5,5), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(512, (3,3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),

    # Block 4
    layers.Conv2D(512, (3,3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),

    # Flatten
    layers.Flatten(),

    # Dense Layers
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    # Output Layer
    layers.Dense(num_classes, activation="softmax")
])


model.compile(
    optimizer=Adam(learning_rate=learn_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

experiment_name = "FER2013-Emotion-Recognition"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_params({
        "img_height": img_height,
        "img_width": img_width,
        "channels": img_channels,
        "batch_size": batch_size,
        "epochs": epoch,
        "learning_rate": learn_rate,
        "classes": classes
    })

    history=model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epoch,
        # class_weight='balanced',  # Handle class imbalance
        callbacks=[
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
    )
    


    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Model Accuracy')
    mlflow.log_artifact("accuracy_plot.png")

    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Model Loss')
    mlflow.log_artifact("loss_plot.png")
    model.save(model_save_dir)

    mlflow.log_artifact(model_save_dir)