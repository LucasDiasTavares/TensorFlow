import pdb

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten

IMAGE_SIZE = [224, 224]
TrainFolder = "data/train"
ValidationFolder = "data/train"

result = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in result.layers:
    layer.trainable = False  # dont need train. It's already trained model

car_names = glob(f"{ValidationFolder}/*")
print(f"QTD car_names: {len(car_names)}")

FlattenLayer = Flatten()(result.output)

# Add Dense layer with car_names
prediction = Dense(len(car_names), activation='softmax')(FlattenLayer)

# Model
model = Model(inputs=result.input, outputs=prediction)
print(f"Model Summary: {model.summary()}")

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

# Augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory(TrainFolder, target_size=(224, 224), batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(ValidationFolder, target_size=(224, 224), batch_size=32,
                                            class_mode='categorical')

result_model_fit = model.fit(training_set, validation_data=test_set, epochs=50, steps_per_epoch=len(training_set))

# Accuracy
plt.plot(result.history['accuracy'], label='train_accuracy')
plt.plot(result.history['value_accuracy'], label='value_accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(result.history['loss'], label='train_loss')
plt.plot(result.history['value_loss'], label='value_loss')
plt.legend()
plt.show()

# Save Model
model.save('car_brands.h5')
