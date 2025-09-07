import numpy as np
import cv2
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# User configuration
DATASET_PATH = Path("./dataset_opencv")
MODEL_NAME = 'trackmania_model_v2.h5'
IMG_HEIGHT = 90
IMG_WIDTH = 160
IMG_CHANNELS = 3
EPOCHS = 50
BATCH_SIZE = 32

def augment_image(image, action):
    """
    Applies random augmentation to an image and its corresponding action.
    """
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        a_key = action[1]
        d_key = action[3]
        action[1] = d_key
        action[3] = a_key

    if random.random() > 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ratio = 0.5 + random.random()
        hsv[:,:,2] = np.clip(hsv[:,:,2] * ratio, 0, 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    return image, action

def load_data(dataset_path):
    """
    Loads the dataset and applies augmentation to the training data.
    """
    print(f"Loading data from {dataset_path}...")
    
    image_files = sorted(dataset_path.glob("frame_*.png"))
    action_files = sorted(dataset_path.glob("action_*.npy"))

    if not image_files or not action_files:
        raise FileNotFoundError(f"No data found in {dataset_path}. Did you run the data collector?")

    assert len(image_files) == len(action_files), "Mismatch between number of images and actions."

    images = []
    actions = []

    for img_path, act_path in zip(image_files, action_files):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        action = np.load(act_path)
        
        aug_img, aug_action = augment_image(img.copy(), action.copy())

        images.append(img)
        actions.append(action)
        images.append(aug_img)
        actions.append(aug_action)

    print(f"Loaded and augmented {len(images)} samples.")
    
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(actions, dtype=np.float32)

    return X, y

def build_model():
    """
    Builds the improved CNN architecture with Batch Normalization.
    """
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        BatchNormalization(),
        
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(),
        
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        
        Flatten(),
        
        Dense(100, activation='relu'),
        Dropout(0.3),
        
        Dense(50, activation='relu'),
        Dropout(0.3),
        
        Dense(10, activation='relu'),
        
        Dense(4, activation='sigmoid')  
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_history(history):
    """
    Plots the training and validation loss and accuracy.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    print(history.history['accuracy'])
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nSaved training history plot to 'training_history.png'")
    plt.show()

def main():
    """Main function to load data, build model, and start training."""
    
    try:
        X, y = load_data(DATASET_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    model = build_model()
    model.summary()
    
    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, mode='min', verbose=1)

    print("\n--- Starting Training ---")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    print("\n--- Training Finished ---")
    print(f"The best model has been saved as '{MODEL_NAME}'")
    
    plot_history(history)

if __name__ == "__main__":
    main()
