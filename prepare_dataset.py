import os
import cv2

# Define the new data directory path
DATA_DIR = r"C:\Users\user\Downloads\ASL_dataset"

# Create the main data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36  # For A-Z
dataset_size = 100  # Number of samples per class

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for class {j}. Press 's' to start capturing...")

    # Wait until user presses 's' to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access the camera. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.putText(frame, f"Press 's' to start class {j} samples", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Start when 's' is pressed
            break
        elif key == ord('q'):  # Quit program
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Start collecting dataset_size number of images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.imshow("Frame", frame)
        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
        counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit if 'q' is pressed
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"Class {j} data collection complete!")

print("All 26 classes collected. Process finished!")

cap.release()
cv2.destroyAllWindows()
