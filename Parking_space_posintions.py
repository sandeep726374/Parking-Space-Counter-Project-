import cv2
import pickle

# Load the first frame of the video (or any image of the parking lot)
img = cv2.imread("C:/Users/sande/OneDrive/Desktop/Parking Space Counter/Parking-Space-Counter-using-OpenCV-Python-Computer-Vision/carParkImg.png")  # Use full path

# Check if image loaded successfully
if img is None:
    print("Error: Unable to load image.")
    exit()

# List to hold the positions
pos_list = []

# Width and height of each parking spot box
width, height = 107, 48

# Mouse click function to draw rectangles
def mouse_click(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        pos_list.append((x, y))
    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(pos_list):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                pos_list.pop(i)
                break
    with open("positions", "wb") as f:
        pickle.dump(pos_list, f)

# Set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_click)

# Loop to show and draw rectangles
while True:
    img_copy = img.copy()
    for pos in pos_list:
        cv2.rectangle(img_copy, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)

    cv2.imshow("Image", img_copy)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
