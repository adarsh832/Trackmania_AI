import cv2
import numpy as np
import mss
from pynput import keyboard
from pathlib import Path
import time

# User configuration
MONITOR_CONFIG = {"top": 393, "left": 0, "width": 1920, "height": 687}
ACTION_KEYS = ['w', 'a', 's', 'd']
STOP_KEY = keyboard.Key.esc
DATASET_PATH = Path("../dataset_opencv")

pressed_keys = set()
running = True

def on_key_press(key):
    """Callback function for when a key is pressed."""
    global pressed_keys, running
    
    if key == STOP_KEY:
        print("\nStop key pressed. Exiting...")
        running = False
        return False

    try:
        key_char = key.char.lower()
        if key_char in ACTION_KEYS:
            pressed_keys.add(key_char)
    except AttributeError:
        pass

def on_key_release(key):
    """Callback function for when a key is released."""
    global pressed_keys
    try:
        key_char = key.char.lower()
        if key_char in pressed_keys:
            pressed_keys.remove(key_char)
    except (AttributeError, KeyError):
        pass

def create_action_array():
    """Creates a numpy array representing the current state of pressed keys."""
    return np.array([1.0 if key in pressed_keys else 0.0 for key in ACTION_KEYS], dtype=np.float32)

def main():
    """Main function to run the data collection."""
    
    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.start()

    print("--- OpenCV Data Collector V2 ---")
    print(f"Data will be saved in: {DATASET_PATH}")
    print("\nStarting in 5 seconds. Please switch to the TrackMania window.")
    
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
        
    print("\n--- RECORDING STARTED ---")
    print(f"Press the '{STOP_KEY.name.upper()}' key to stop.")

    frame_counter = 0
    start_time = time.time()

    with mss.mss() as sct:
        while running:
            screen_grab = sct.grab(MONITOR_CONFIG)
            frame = np.array(screen_grab)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            action = create_action_array()

            image_path = DATASET_PATH / f"frame_{frame_counter:05d}.png"
            action_path = DATASET_PATH / f"action_{frame_counter:05d}.npy"
            cv2.imwrite(str(image_path), frame_bgr)
            np.save(action_path, action)
            
            frame_counter += 1
            
            if frame_counter % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_counter / elapsed_time
                print(f"Recorded {frame_counter} frames... (FPS: {fps:.2f})")
    
    listener.stop()
    print("\n--- RECORDING STOPPED ---")
    print(f"Total frames saved: {frame_counter}")
    if frame_counter == 0:
        print("Warning: No data was saved. The script may have been stopped too quickly.")

if __name__ == "__main__":
    main()