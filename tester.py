import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Define constants
TEST_DIR = './pred'
IMG_SIZE = 64

# Load the saved model
model = load_model('brain_tumor_classifier.h5')

# Load and preprocess the test images
test_images = []
test_filenames = []

# Load the test images
for filename in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, filename)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    img_array = image.img_to_array(img)
    test_images.append(img_array)
    test_filenames.append(filename)

# Convert the test images to numpy array
test_images = np.array(test_images) / 255.0

# Reshape the test images to match the expected input shape of the model
test_images = np.expand_dims(test_images, axis=-1)

# Make predictions on the test images
predictions = model.predict(test_images)

# Interpret the predictions
for i in range(len(predictions)):
    if predictions[i] > 0.5:
        print(f"Image: {test_filenames[i]} - Tumor Detected")
    else:
        print(f"Image: {test_filenames[i]} - No Tumor Detected")
