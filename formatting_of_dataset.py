import os
import pickle
import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Paths
DATA_DIR = r"C:\Users\user\Downloads\ASL_dataset"
SAVE_PATH = r"C:\Users\user\Downloads\data.pickle"

data, labels = [], []
valid_extensions = {".jpg", ".jpeg", ".png"}

# Process images
for label in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label)
    
    if not os.path.isdir(folder_path) or not label.isdigit():
        continue  # Skip non-numeric folders
    
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(tuple(valid_extensions)):
            continue  # Skip non-image files

        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable images

        # Process image with Mediapipe
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            continue  # Skip if no hands detected

        # Extract hand landmarks
        x_, y_, data_aux = [], [], []
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        data.append(data_aux)
        labels.append(label)

# Save data
with open(SAVE_PATH, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Processing complete!")
