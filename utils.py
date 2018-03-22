import cv2
import dlib
import numpy as np
import pygame

from constants import PREDICTOR_PATH, SCREEN

predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_cascade = cv2.CascadeClassifier('shared/haarcascade_frontalface_default.xml')


# face_cascade = cv2.CascadeClassifier('shared/haarcascade_frontalface_alt2.xml')
# face_cascade = cv2.CascadeClassifier('shared/haarcascade_frontalface_alt.xml')
# eye_cascade = cv2.CascadeClassifier('shared/haarcascade_eye.xml')


def find_faces(im_gray, im_annotated):
    faces = face_cascade.detectMultiScale(
        im_gray,
        scaleFactor=1.3,
        minNeighbors=6,
        minSize=(10, 10)
    )
    # eyes = eye_cascade.detectMultiScale(im_gray)
    # added in eye detection to improve detection accuracy. This might
    # be too slow

    if len(faces) != 0:
        face_found = 1
        print "Found {0} faces!".format(len(faces))
    else:
        face_found = 0

    for (x, y, w, h) in faces:
        # draw rectangle around faces
        cv2.rectangle(im_annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return faces, im_annotated[0:], face_found


def get_landmarks(im, roi):
    x = roi[0][0]
    y = roi[0][1]
    w = roi[0][2]
    h = roi[0][3]
    dlib_rect = dlib.rectangle(long(x), long(y), long(x + w), long(y + h))
    detected_landmarks = predictor(im, dlib_rect).parts()
    # get the points for the landmarks
    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
    return landmarks


def bbox_get(roi):
    bbox = roi[0][0], roi[0][1], roi[0][2], roi[0][3]
    bbox = np.asarray(bbox)
    return bbox


def biggest_face(roi):
    if roi.shape[0] > 1:
        maxWidth = roi[0][3]
        for _ in roi:
            if _[3] >= maxWidth:
                roi_new = _

        if roi_new.size != 0:
            roi = roi_new
        else:
            roi = roi[0]
    return roi


def resize_ff(im):
    """This function takes an input image, scales it to fill the window while
    maintaining aspect ratio and then adds a border in the horizontal dimension"""
    # SCREEN = [900, 1440]  # rows,cols (height, width)

    # rows, cols, ch = im.shape
    # aspect = SCREEN[0] / rows  # make height match fullscreen height
    # x_dim = min(int(aspect * cols),int(SCREEN[1]))
    # y_dim = int(SCREEN[0])
    # # To shrink an image, it will generally look best with CV_INTER_AREA
    # # interpolation, whereas to enlarge an image, it will generally look best
    # # with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK).
    # im = cv2.resize(im, (x_dim, y_dim), 0, 0, interpolation=cv2.INTER_LINEAR)

    # # Add border to resulting image if it isn't wide enough
    rows, cols, ch = im.shape

    width_correct = (SCREEN[1] - cols) / 2
    height_correct = (SCREEN[0] - rows) / 2
    # if height_correct <= 0:
    #     height_correct = 1
    # if height_correct >= 200:
    #     height_correct = 200
    print height_correct

    top = int(height_correct)
    bottom = int(height_correct)
    left = int(width_correct)
    right = int(width_correct)
    borderType = cv2.BORDER_CONSTANT
    colour = (0, 0, 0)

    print im.shape
    im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType, colour)

    return im


def running_mean(x, N):
    """http://stackoverflow.com/questions/13728392/moving-average-or-running-mean"""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N
