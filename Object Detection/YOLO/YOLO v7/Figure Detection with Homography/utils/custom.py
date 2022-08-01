import cv2
import numpy as np


def mouse_handler(event, x, y, flags, data):
    if len(data['points']) > 4:
        data['points'] = []
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['img'], (x, y), 3, (0, 0, 255), 5, 16)
        cv2.imshow("Image", data['img'])
        if len(data['points']) < 5:
            data['points'].append([x, y])


def get_four_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points


def get_four_points_video(video):
    data = {}
    data['video'] = video.get()
    data['points'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", video)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points
