import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image
import pyglet
import math

from pyglet.gl import *
from pyglet.math import Mat4, Vec3

from AR_model import Model

FREIGHT_DISTANCE = 50


INVERSE_MATRIX = np.array([ [ 1.0, 1.0, 1.0, 1.0],
                            [-1.0,-1.0,-1.0,-1.0],
                            [-1.0,-1.0,-1.0,-1.0],
                            [ 1.0, 1.0, 1.0, 1.0]])

## to convert color space of opencv to color space of pyglet
## https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
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


## estimates the position of a marker within the camera coordinate system
## returns rotation and translation vectors of marker in camera coordinate system
def estimatePoseMarker(corners, mtx, distortion):
    global length
    length = abs(corners[0][0][0] - corners[0][1][0]) if (abs(corners[0][0][0] - corners[0][1][0]) > abs(corners[0][0][0] - corners[0][2][0])) else abs(corners[0][0][0] - corners[0][2][0])
    marker_points = np.array([[-length / 2, length / 2, 0],
                              [length / 2, length / 2, 0],
                              [length / 2, -length / 2, 0],
                              [-length / 2, -length / 2, 0]], dtype=np.float32)
    rvecs = []
    tvecs = []
    for c in corners:
        _, r, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(r)
        tvecs.append(t)
    return np.array([rvecs]), np.array([tvecs])


def get_center_of_marker(corners):
    center_x = corners[0][0] - ((corners[0][0] - corners[2][0])/2)
    center_y = corners[0][1] - ((corners[0][1] - corners[2][1])/2)
    return (center_x, center_y)


## window params
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_Z = 420

## setup pyglet image
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, resizable=False)

## setup the camera
cap = cv2.VideoCapture(0)
cameraMatrix = np.array([[534.34144579, 0., 339.15527836], [0., 534.68425882,  233.84359494], [0., 0., 1.]], dtype=np.float64)
distCoeffs = np.array([0, 0, 0, 0], dtype=np.float64)

## setup aruco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

## for animated/moving 3D objects
# time = 0


@window.event
def on_draw():
    global view_matrix, position
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AruCo markers in the frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    aruco.drawDetectedMarkers(frame, corners)

    if ids is not None:
        for i, marker_id in enumerate(ids):
            position = get_center_of_marker(corners[i][0])
            rvec, tvec = estimatePoseMarker(corners[i], cameraMatrix, distCoeffs)

            # convert the rotation vector into a rotation matrix
            # then, make OpenCV matrix compatible with OpenGL 
            rmtx = cv2.Rodrigues(rvec[0])[0]
            view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvec[0,0,0]],
                                    [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvec[0,0,1]],
                                    [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvec[0,0,2]],
                                    [0.0       ,0.0       ,0.0       ,1.0    ]], dtype="object")
            view_matrix = view_matrix * INVERSE_MATRIX
            view_matrix = np.transpose(view_matrix)
            for model in models:
                model.setup_translation(marker_id, view_matrix, position, length)

    img = cv2glet(frame, 'BGR')
    window.clear()
    img.blit(-WINDOW_WIDTH/2, -WINDOW_HEIGHT/2, 0)
    for model in models:
            model.batch.draw()


def animate(dt):
    ## pass a time component for animated/moving 3D objects
    # global time
    # time += dt
    enton_w = 0
    enton_h = 0
    enton_present = False
    fire_w = 0
    fire_h = 0
    fire_present = False
    freight = False
    for model in models:
        if model._id == 4 and model._position is not None:
            enton_w, enton_h = model._position
            enton_present = True
        if model._id == 5 and model._position is not None:
            fire_w, fire_h = model._position
            fire_present = True
    if enton_present and fire_present:
        distance = math.sqrt((fire_w - enton_w)**2 + (fire_h - enton_h)**2)
        if distance <= FREIGHT_DISTANCE:
            freight = True
        else:
            freight = False
    for model in models:
        if model._id == 4:
            if freight:
                model._rot_x += 20
                model._rot_y += 20
                model._rot_z += 20
            else:
                model._rot_x = 0
                model._rot_y = 0
                model._rot_z = 0
        model.animate()    


# not relevant for us for now
@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.1, z_far=1024)
    return pyglet.event.EVENT_HANDLED


if __name__ == "__main__":
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    models = []
    models.append(Model(path="./ar_game_3d/enton.obj", id=4, win_w=WINDOW_WIDTH, win_h=WINDOW_HEIGHT, rot_x=270, rot_y=90, rot_z=270, scaling_factor=0.2))
    models.append(Model(path="./ar_game_3d/fireball.obj", id=5, win_w=WINDOW_WIDTH, win_h=WINDOW_HEIGHT, rot_x=270, rot_y=90, rot_z=270, scaling_factor=2))

    # Set the application wide view matrix (camera):
    window.view = Mat4.look_at(position=Vec3(0, 0, WINDOW_Z), target=Vec3(0, 0, 0), up=Vec3(0, 1, 0))
    window.viewport = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.1, z_far=1024)

    pyglet.clock.schedule_interval(animate, 1 / 60)
    pyglet.app.run()
