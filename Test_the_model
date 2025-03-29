import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# Load the trained model
model_dir = r"C:\Users\user\Downloads\model.p"
if not os.path.exists(model_dir):
    print(f" Error: Model file not found at {model_dir}")
    exit()

try:
    with open(model_dir, 'rb') as file:
        model_dict = pickle.load(file)
    model = model_dict['model']
    expected_features = model.n_features_in_  # Get the number of features expected by the model
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize webcam (Change `0` if needed)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Labels Dictionary (A-Z mapping)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract normalized landmark coordinates
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Ensure feature vector matches the expected size
        if len(data_aux) < expected_features:
            data_aux.extend([0] * (expected_features - len(data_aux)))  # Pad with zeros
        elif len(data_aux) > expected_features:
            print(f" Warning: Trimming feature vector from {len(data_aux)} to {expected_features}.")
            data_aux = data_aux[:expected_features]

        # Predict the ASL character
        prediction = model.predict([np.asarray(data_aux)])
        predicted_class = int(prediction[0])  # Assuming the model outputs class indices
        predicted_character = labels_dict.get(predicted_class, '?')  # Default '?' if out of range

        # Calculate bounding box coordinates
        x1, y1 = max(0, int(min(x_) * W) - 10), max(0, int(min(y_) * H) - 10)
        x2, y2 = min(W, int(max(x_) * W) + 10), min(H, int(max(y_) * H) + 10)

        # Draw bounding box and predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('ASL Detection', frame)

    # Exit loop when "Esc" key (ASCII 27) is pressed
    if cv2.waitKey(1) == 27:
        print("Esc key pressed! Closing camera...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
