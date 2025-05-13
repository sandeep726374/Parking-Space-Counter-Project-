import cv2
import numpy as np
import pickle
import datetime
import os
import csv
import time
import pygame  # Import pygame for sound functionality

# Initialize pygame mixer
pygame.mixer.init()

# Load buzzer sound (Make sure to have a buzzer sound file like 'buzzer.wav' in your working directory)
buzzer_sound = pygame.mixer.Sound("C:/Users/sande/OneDrive/Desktop/Parking Space Counter/Parking-Space-Counter-using-OpenCV-Python-Computer-Vision/buzzer.wav")

# Load video
cap = cv2.VideoCapture("C:/Users/sande/OneDrive/Desktop/Parking Space Counter/Parking-Space-Counter-using-OpenCV-Python-Computer-Vision/carPark.mp4")

if not cap.isOpened():
    print("Error: Unable to load video.")
    exit()

# Load parking positions
try:
    with open("positions", "rb") as file:
        pos_list = pickle.load(file)
except FileNotFoundError:
    print("[ERROR] 'positions' file not found. Run the positioning script first.")
    exit()

# Parameters
width, height = 107, 48
font = cv2.FONT_HERSHEY_TRIPLEX  # Times New Roman like
prev_status = ["free"] * len(pos_list)
last_log_time = time.time()

# Prepare CSV logging
if not os.path.exists("parking_log.csv"):
    with open("parking_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Total", "Occupied", "Free"])

# Initialize buzzer counter and a flag for buzzer state
buzzer_counter = 0
max_buzzer_count = 3  # Max number of times buzzer will play
last_filled_state = False  # Keep track of the last state if parking was filled

# Draw pie chart on frame (Modified to add 50px top padding and reduce size)
def draw_pie_chart(frame, occupied, total):
    center = (frame.shape[1] - 80, 90)  # 50px top padding added to the Y-coordinate
    radius = 40  # Reduced size
    free = total - occupied
    free_angle = int((free / total) * 360)
    cv2.circle(frame, center, radius, (200, 200, 200), -1)
    cv2.ellipse(frame, center, (radius, radius), 0, 0, free_angle, (0, 255, 0), -1)
    cv2.ellipse(frame, center, (radius, radius), 0, free_angle, 360, (0, 0, 255), -1)
    cv2.circle(frame, center, radius, (0, 0, 0), 2)
    cv2.putText(frame, "Free", (frame.shape[1] - 130, 145), font, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Occupied", (frame.shape[1] - 130, 165), font, 0.5, (0, 0, 255), 1)

# Log data to CSV
def log_data(total, occupied, available):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("parking_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, total, occupied, available])

# Check and mark spaces
def check_parking_space(img_processed, frame):
    global prev_status, last_log_time, buzzer_counter, last_filled_state
    available_places = 0
    occupied_places = 0

    for i, pos in enumerate(pos_list):
        x, y = pos
        img_crop = img_processed[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)

        status = "occupied" if count > 750 else "free"
        if status == "free":
            available_places += 1
            color = (0, 255, 0)
        else:
            occupied_places += 1
            color = (0, 0, 255)

        # Save snapshot when car arrives
        if prev_status[i] == "free" and status == "occupied":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"slot_{i+1}_{timestamp}.jpg", frame[y:y + height, x:x + width])

        prev_status[i] = status

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(frame, f"Slot {i+1}", (x + 2, y + 20), font, 0.5, (255, 255, 255), 1)

    total = len(pos_list)
    cv2.putText(frame, f"Total Parking Slots: {total}", (50, 50), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Parking Slots Occupied: {occupied_places}", (50, 90), font, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Parking Slots Free: {available_places}", (50, 130), font, 0.7, (0, 255, 0), 2)

    draw_pie_chart(frame, occupied_places, total)

    # Trigger buzzer sound when all slots are filled and play only 3 times
    if occupied_places == total and not last_filled_state:
        if buzzer_counter < max_buzzer_count:
            buzzer_sound.play()  # Play buzzer sound
            buzzer_counter += 1  # Increment buzzer counter
        last_filled_state = True  # Set the flag to True when the parking is filled

    # If the parking slots become free, reset the buzzer counter and flag
    if occupied_places < total:
        last_filled_state = False

    # Log every 5 seconds
    if time.time() - last_log_time >= 5:
        log_data(total, occupied_places, available_places)
        last_log_time = time.time()

# Main Loop
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)
    median = cv2.medianBlur(thresh, 5)
    dilated = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)

    # Timestamp (Moved left by 50px)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Time: {time_str}", (900 - 50, 40), font, 0.5, (255, 255, 0), 1)

    check_parking_space(dilated, frame)

    cv2.imshow("Unique Parking Monitor", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
