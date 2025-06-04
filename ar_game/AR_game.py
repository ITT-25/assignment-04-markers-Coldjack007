import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
import sys
import os
import random

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

#Used for extracting the inner portion of the AruCo board
corners_unsorted = []
inner_corners = []

first_frame = cv2.imread("./ar_game/start_screen.png")
last_frame = first_frame
ending_message = pyglet.text.Label("Game Over! Press P to restart and Q to quit!", WINDOW_WIDTH/2, WINDOW_HEIGHT/2, anchor_x="center", anchor_y="center", font_name="Times New Roman", font_size=24)

finger_position = [0, 0]

# Define the ArUco dictionary, parameters, and detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

game_active = False
game_over = False
game_score = 0

#Bubble Params
BUBBLE_RADIUS = 1
BUBBLE_COLOR = (255,0,0)
BUBBLE_MAX_RADIUS = 200
batch = pyglet.graphics.Batch()
bubble_array = []
BUBBLE_ARRAY_MAX_NUM = 25
BUBBLE_SPAWN_PERCENTAGE = 2 #out of 100
current_acceleration = 1
speedup_counter = 0
SPEEDUP_THRESHOLD = 8000

class Enemy_Bubble:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.body = pyglet.shapes.Circle(x, y, BUBBLE_RADIUS, color=BUBBLE_COLOR, batch=batch)


# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

#Detects the AruCos on the board, and reacts accordingly if all four AruCos are present
def catch_arucos(frame):
    global corners_unsorted, last_frame, game_active
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    # Check if marker is detected
    if ids is not None:
        if len(ids) == 4:
            # Draw lines along the sides of the marker
            #aruco.drawDetectedMarkers(frame, corners)
            corners_unsorted = corners
            extract_inner_corners()
            order_points()
            frame = warp_picture(frame)
            last_frame = frame
            game_active = True
            return frame
    return last_frame

#From the four AruCos, extract the four innermost points.
def extract_inner_corners():
    global corners_unsorted, inner_corners
    all_corners = np.vstack([m[0] for m in corners_unsorted])
    center_point = np.mean (all_corners, axis=0)
    inner_corners = []
    for corner in corners_unsorted:
        current_corner = corner[0]
        distance = np.linalg.norm(current_corner - center_point, axis=1)
        closest_corner = np.argmin(distance)
        inner_corners.append(current_corner[closest_corner])
    inner_corners = np.array(inner_corners, dtype="float32")

#Orders four given points to top-left, top-right, bottom-right, bottom-left
def order_points():
    global inner_corners
    sum = inner_corners.sum(axis=1)
    diff = np.diff(inner_corners, axis=1)

    inner_corners = np.array([
        inner_corners[np.argmin(sum)],
        inner_corners[np.argmin(diff)],
        inner_corners[np.argmax(sum)],
        inner_corners[np.argmax(diff)]
    ], dtype="float32")

#Warps the image with the four given points
def warp_picture(frame):
    global inner_corners
    rectangle_points = np.array([
        [0,0],
        [WINDOW_WIDTH - 1, 0],
        [WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1],
        [0, WINDOW_HEIGHT - 1]
    ], dtype="float32")
    perspectiveTransform = cv2.getPerspectiveTransform(inner_corners, rectangle_points)
    return cv2.warpPerspective(frame, perspectiveTransform, (WINDOW_WIDTH, WINDOW_HEIGHT))

#Detects the hand based on common skin tone ranges. Returns a mask of the hand.
def get_hand_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = (0, 20, 70)
    upper_skin = (20, 255, 255)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

#The largest contour of the hand mask will be the entire hand.
def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

#Returns the highest point of the hand contour, in most cases a fingertip.
def get_fingertip_point(contour):
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    return topmost

def capture_finger(frame):
    global finger_position
    mask = get_hand_mask(frame)
    contour = get_largest_contour(mask)
    if contour is not None and cv2.contourArea(contour) > 1000:
            fingertip = get_fingertip_point(contour)
            finger_position = [fingertip[0], fingertip[1]]


#Makes the bubbles grow, checks Game Over Condition, accelerates the game if necessary
def progress_game():
    global bubble_array, speedup_counter, current_acceleration
    #Determine whether a new EnemyBubble should spawn
    create_bubbles()
    #Let all bubbles grow, if one pops, game over
    if bubble_array:
        for bubble in bubble_array:
            bubble.body.radius += current_acceleration
            speedup_counter += 1
            if speedup_counter == SPEEDUP_THRESHOLD:
                speedup_counter = 0
                current_acceleration += 1
            if bubble.body.radius >= BUBBLE_MAX_RADIUS:
                end_game()

#Creates one or more new EnemyBubbles
def create_bubbles():
    if len(bubble_array) < BUBBLE_ARRAY_MAX_NUM:
        if random.randint(1,100) <= BUBBLE_SPAWN_PERCENTAGE or len(bubble_array) == 0:
            new_bubble = Enemy_Bubble(random.randint(0,WINDOW_WIDTH), random.randint(0,WINDOW_HEIGHT))
            bubble_array.append(new_bubble)
            create_bubbles() #chance of spawning multiple bubbles at once

#Game Over
def end_game():
    global game_active, game_over, game_score, bubble_array
    bubble_array.clear()
    game_active = False
    game_over = True
    print("Your score is: " + str(game_score))

#compute collision between contours and bubbles.
def compute_input():
    global bubble_array, game_score
    new_bubble_array = []
    correct_finger_pos = (finger_position[0], WINDOW_HEIGHT - finger_position[1])
    for bubble in bubble_array:
        dx = correct_finger_pos[0] - bubble.x
        dy = correct_finger_pos[1] - bubble.y
        if dx**2 + dy**2 > bubble.body.radius**2:
            new_bubble_array.append(bubble)
        else:
            game_score += 1
    bubble_array = new_bubble_array

#Back to the starting menu
def exit_game():
    global last_frame, game_over, game_active, game_score
    last_frame = first_frame
    game_over = False
    game_active = False
    game_score = 0


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        os._exit(0)
    if symbol == pyglet.window.key.P:
        exit_game()

@window.event
def on_draw():
    window.clear()
    if game_over == False:
        ret, frame = cap.read()
        frame = catch_arucos(frame)
        capture_finger(frame)
        if game_active:
            progress_game()
            compute_input()
        img = cv2glet(frame, 'BGR')
        img.blit(0, 0, 0)
        batch.draw()
        finger_circle = pyglet.shapes.Circle(finger_position[0], WINDOW_HEIGHT - finger_position[1], 10, color=(0, 255, 0))
        finger_circle.draw()
    else:
        ending_message.draw()

pyglet.app.run()
