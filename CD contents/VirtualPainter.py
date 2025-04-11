import mediapipe as mp
import cv2
import numpy as np
import time

# Constants
ml = 150  # Margin left for tools
max_x, max_y = 250 + ml, 50  # Dimensions for tool selection area
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Color palette
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255)   # Cyan
]
color_index = 0
color_box_size = 40  # Size of each color box
color_palette = np.zeros((color_box_size, len(colors) * color_box_size, 3), dtype=np.uint8)

# Initialize color palette
for i, color in enumerate(colors):
    color_palette[:, i * color_box_size:(i + 1) * color_box_size] = color

# FPS variables
fps_start_time = 0
fps_frame_count = 0
fps = 0

# Smoothing variables
smooth_points = []  # Store previous points for smoothing

# Get tools function
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

# Get color function
def getColor(x):
    global color_index
    # Calculate which color box the x-coordinate falls into
    color_index = (x - (ml + 250)) // color_box_size
    if color_index < 0:
        color_index = 0
    elif color_index >= len(colors):
        color_index = len(colors) - 1
    return colors[color_index]

# Check if index finger is raised
def index_raised(yi, y9):
    return (y9 - yi) > 40

# Check if fist is detected (including thumb)
def is_fist(landmarks):
    # Landmarks for fingertips (4, 8, 12, 16, 20) and their corresponding knuckles (2, 6, 10, 14, 18)
    fingertip_indices = [4, 8, 12, 16, 20]
    knuckle_indices = [2, 6, 10, 14, 18]
    threshold = 20  # Threshold to determine if a finger is closed

    for ft, kn in zip(fingertip_indices, knuckle_indices):
        fingertip_y = landmarks[ft].y * 480
        knuckle_y = landmarks[kn].y * 480
        if abs(fingertip_y - knuckle_y) > threshold:
            return False  # Finger is not closed
    return True  # All fingers are closed (fist detected)

# Linear interpolation between two points
def interpolate_points(p1, p2, num_points=10):
    return list(zip(
        np.linspace(p1[0], p2[0], num_points, dtype=int),
        np.linspace(p1[1], p2[1], num_points, dtype=int)
    ))

# Initialize MediaPipe Hands
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

# Initialize mask as a 3-channel image
mask = np.ones((480, 640, 3), dtype=np.uint8) * 255

cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    # Calculate FPS
    if fps_frame_count == 0:
        fps_start_time = time.time()
    fps_frame_count += 1
    if fps_frame_count >= 10:  # Update FPS every 10 frames
        fps_end_time = time.time()
        fps = int(fps_frame_count / (fps_end_time - fps_start_time))
        fps_frame_count = 0

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            landmarks = i.landmark
            x, y = int(landmarks[8].x * 640), int(landmarks[8].y * 480)

            # Tool selection
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Your current tool set to:", curr_tool)
                    time_init = True
                    rad = 40

            else:
                time_init = True
                rad = 40

            # Color selection
            if ml + 250 < x < ml + 250 + len(colors) * color_box_size and max_y + 10 < y < max_y + 10 + color_box_size:
                getColor(x)
                # Highlight the selected color box
                cv2.rectangle(frm, (ml + 250 + color_index * color_box_size, max_y + 10),
                              (ml + 250 + (color_index + 1) * color_box_size, max_y + 10 + color_box_size),
                              (255, 255, 255), 2)
                # Display the selected color name
                cv2.putText(frm, f"Selected Color: {colors[color_index]}", (ml + 250, max_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_index], 2)

            # Drawing tools
            if curr_tool == "draw":
                xi, yi = int(landmarks[12].x * 640), int(landmarks[12].y * 480)
                y9 = int(landmarks[9].y * 480)

                if index_raised(yi, y9):
                    if len(smooth_points) > 0:
                        # Interpolate between the last point and the current point
                        for p in interpolate_points((prevx, prevy), (x, y)):
                            cv2.line(mask, (prevx, prevy), p, colors[color_index], thick)
                            prevx, prevy = p
                    else:
                        cv2.line(mask, (prevx, prevy), (x, y), colors[color_index], thick)
                    smooth_points.append((x, y))  # Add current point to the list
                else:
                    prevx, prevy = x, y
                    smooth_points = []  # Reset points when not drawing

            elif curr_tool == "line":
                xi, yi = int(landmarks[12].x * 640), int(landmarks[12].y * 480)
                y9 = int(landmarks[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), colors[color_index], thick)
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), colors[color_index], thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                xi, yi = int(landmarks[12].x * 640), int(landmarks[12].y * 480)
                y9 = int(landmarks[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y), colors[color_index], thick)
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), colors[color_index], thick)
                        var_inits = False

            elif curr_tool == "circle":
                xi, yi = int(landmarks[12].x * 640), int(landmarks[12].y * 480)
                y9 = int(landmarks[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), colors[color_index], thick)
                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), colors[color_index], thick)
                        var_inits = False

            elif curr_tool == "erase":
                xi, yi = int(landmarks[12].x * 640), int(landmarks[12].y * 480)
                y9 = int(landmarks[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, (255, 255, 255), -1)

            # Fist gesture detection (including thumb)
            if is_fist(landmarks):
                mask = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Clear the board
                cv2.putText(frm, "Board Cleared!", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display color palette
    frm[max_y + 10:max_y + 10 + color_box_size, ml + 250:ml + 250 + len(colors) * color_box_size] = color_palette

    # Combine mask and frame
    frm = cv2.bitwise_and(frm, mask)

    # Display tools
    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

    # Display current tool and selected color
    cv2.putText(frm, f"Tool: {curr_tool}", (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frm, f"Color: {colors[color_index]}", (270 + ml, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[color_index], 2)

    # Display FPS
    cv2.putText(frm, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Air Canvas", frm)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        cv2.destroyAllWindows()
        cap.release()
        break