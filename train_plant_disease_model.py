import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# CONFIG
IMAGE_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 32
MAX_IMAGES_PER_CLASS = 500
DATASET_DIR = r"C:\Users\LENOVO\OneDrive\Documents\inpro\PlantVillage"  # Update as needed
MODEL_PATH = "best_model.h5"
LABEL_PATH = "class_names.txt"

# Load dataset
images, labels = [], []
class_names = sorted(os.listdir(DATASET_DIR))
label_map = {name: i for i, name in enumerate(class_names)}

print("🔄 Loading images...")
for class_name in class_names:
    class_path = os.path.join(DATASET_DIR, class_name)
    count = 0
    for file in os.listdir(class_path):
        if count >= MAX_IMAGES_PER_CLASS:
            break
        img_path = os.path.join(class_path, file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(label_map[class_name])
            count += 1
        except:
            continue

# Prepare data
X = np.array(images, dtype='float32') / 255.0
y = to_categorical(labels, num_classes=len(class_names))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Loaded {len(X_train)} training and {len(X_val)} validation images.")

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("🚀 Training the model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model and labels
model.save(MODEL_PATH)
with open(LABEL_PATH, "w") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"🎉 Training complete! Model saved to '{MODEL_PATH}'")
print(f"📄 Class names saved to '{LABEL_PATH}'")
