import cv2
import sys
import numpy as np

selected_points = []
select_mode = False
warped_picture = None

#CMD LINE PARAMS
image_file_path = input("Gib hier bitte eine Bild-Datei an (Darg&Drop gerne ein Bild rein): ")
img = cv2.imread(image_file_path)
save_path = input("Gib hier bitte einen Speicher-Pfad an (gerne auch leerlassen): ")
if save_path != "":
    save_path = save_path + "/"
res_x = input("Gib hier bitte die horizontale Auflösung als Zahl an: ")
res_x = int(res_x)
res_y = input("Gib hier bitte die vertikale Auflösung als Zahl an: ")
res_y = int(res_y)
print("Willkommen! Drücke ESC, um das Bild zu resetten. Drücke S, um das ausgeschnittene Bild zu speichern. Schließe das Fenster, um das programm zu beenden.")

WINDOW_NAME = 'Preview Window'

cv2.namedWindow(WINDOW_NAME)

def mouse_callback(event, x, y, flags, param):
    global img, selected_points, select_mode

    if event == cv2.EVENT_LBUTTONDOWN and select_mode == False:
        img = cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        selected_points.append((x,y))
        cv2.imshow(WINDOW_NAME, img)
        if len(selected_points) == 4:
            select_mode = True
            warp_picture()

#Warps the image with the four given points
def warp_picture():
    global warped_picture
    ordered_points = order_points()
    rectangle_points = np.array([
        [0,0],
        [res_x - 1, 0],
        [res_x - 1, res_y - 1],
        [0, res_y - 1]
    ], dtype="float32")
    perspectiveTransform = cv2.getPerspectiveTransform(ordered_points, rectangle_points)
    warped_picture = cv2.warpPerspective(img, perspectiveTransform, (res_x, res_y))

    cv2.imshow("Warped Picture", warped_picture)

#Orders four given points to top-left, top-right, bottom-right, bottom-left
def order_points():
    points = np.array(selected_points, dtype="float32")
    sum = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    return np.array([
        points[np.argmin(sum)],
        points[np.argmin(diff)],
        points[np.argmax(sum)],
        points[np.argmax(diff)]
    ], dtype="float32")

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
cv2.imshow(WINDOW_NAME, img)

while True:
    #ESC = 27, s = 115, exit = -1
    key = cv2.waitKey(0)
    if key == 27: #Escape, start over
        select_mode = False
        selected_points.clear()
        img = cv2.imread(image_file_path)
        cv2.imshow(WINDOW_NAME, img)
    if key == 115 and select_mode == True: #s, Save the image
        if cv2.imwrite(save_path + "warped_image.png", warped_picture):
            print("Speichern erfolgreich!")
        else:
            print("Hoppla, da ist was schiefgegangen!")
        cv2.destroyWindow("Warped Picture")
        select_mode = False
        selected_points.clear()
        img = cv2.imread(image_file_path)
        cv2.imshow(WINDOW_NAME, img)
    if key == -1: #for exiting the program
        sys.exit()
