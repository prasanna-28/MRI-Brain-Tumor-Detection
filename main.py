import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

# Define constants
TRAIN_DIR = './'
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 40

# Load and label the training images manually
images = []
labels = []

# Load brain tumor images labeled as 'yes'
yes_dir = os.path.join(TRAIN_DIR, 'yes')
yes_images = os.listdir(yes_dir)
for img_name in yes_images:
    img_path = os.path.join(yes_dir, img_name)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    img_array = image.img_to_array(img)
    images.append(img_array)
    labels.append(1)  # Label as 1 for brain tumor

# Load non-tumor images labeled as 'no'
no_dir = os.path.join(TRAIN_DIR, 'no')
no_images = os.listdir(no_dir)
for img_name in no_images:
    img_path = os.path.join(no_dir, img_name)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    img_array = image.img_to_array(img)
    images.append(img_array)
    labels.append(0)  # Label as 0 for no brain tumor

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the pixel values
images = images / 255.0

# Initialize CNN
model = Sequential()
model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN on the manually loaded data
model.fit(x=images, y=labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save the model
model.save('brain_tumor_classifier.h5')
