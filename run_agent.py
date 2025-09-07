import cv2
import numpy as np
import mss
import pydirectinput
import tensorflow as tf
from pathlib import Path
import time

# User configuration
MONITOR_CONFIG = {"top": 393, "left": 0, "width": 1920, "height": 687}
ACTION_KEYS = ['w', 'a', 's', 'd']
MODEL_PATH = Path("./models/trackmania_mode.h5")
IMG_HEIGHT = 90
IMG_WIDTH = 160

def press_keys(action_predictions):
    """
    Simulates key presses based on the model's output predictions.
    A threshold is used to decide whether to press or release a key.
    """
    threshold = 0.85
    
    for i, key in enumerate(ACTION_KEYS):
        if action_predictions[i] > threshold:
            pydirectinput.keyDown(key)
        else:
            pydirectinput.keyUp(key)

def run_agent():
    """Main function to run the AI agent."""
    print(f"Loading trained model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please make sure you have successfully run train_model.py")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    print("Starting AI in 5 seconds. Click on the TrackMania window now!")
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    
    print("\n--- AI IS NOW DRIVING ---")
    print("Press Ctrl+C in this terminal to stop the agent.")

    last_time = time.time()
    
    try:
        with mss.mss() as sct:
            while True:
                screen_grab = sct.grab(MONITOR_CONFIG)
                
                if screen_grab is None:
                    print("Screen grab failed. Skipping frame.")
                    continue
                
                frame = np.array(screen_grab)
                if frame.size == 0:
                    print("Empty frame captured. Skipping frame.")
                    continue

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                processed_frame = cv2.resize(frame_bgr, (IMG_WIDTH, IMG_HEIGHT))
                processed_frame = processed_frame / 255.0
                
                model_input = np.expand_dims(processed_frame, axis=0)

                prediction = model.predict(model_input, verbose=0)[0]

                press_keys(prediction)
                
                fps = 1 / (time.time() - last_time)
                last_time = time.time()
                print(f"FPS: {int(fps)}", end='\r')

    except KeyboardInterrupt:
        print("\nStopping agent...")

    finally:
        for key in ACTION_KEYS:
            pydirectinput.keyUp(key)
        print("\n--- AI STOPPED ---")

if __name__ == "__main__":
    run_agent()